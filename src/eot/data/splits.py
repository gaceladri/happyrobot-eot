"""Deterministic grouped train/dev splits.

Hold out *whole sources* for dev so a high score cannot come from recognising a voice or a TTS
vocoder. With a single source, fall back to ``clip_id`` so prefixes of one clip never straddle
train/dev. A guardrail prevents a dominant source from consuming the whole validation budget.
"""
from __future__ import annotations
import random
from collections import Counter, defaultdict


def split_diagnostics(
    train: list[dict], dev: list[dict], *, group_key: str, requested_group_key: str | None = None
) -> dict:
    """Summarize split balance and prove that groups do not cross the boundary."""
    def describe(part: list[dict]) -> dict:
        labels = Counter(int(row["label"]) for row in part)
        return {
            "n": len(part),
            "labels": {str(k): labels.get(k, 0) for k in (0, 1)},
            "groups": len({str(row.get(group_key, "unknown")) for row in part}),
        }

    train_groups = {str(row.get(group_key, "unknown")) for row in train}
    dev_groups = {str(row.get(group_key, "unknown")) for row in dev}
    return {
        "group_key": group_key,
        "requested_group_key": requested_group_key or group_key,
        "train": describe(train),
        "dev": describe(dev),
        "group_overlap": sorted(train_groups & dev_groups),
    }


def grouped_split(
    rows: list[dict],
    dev_frac: float = 0.15,
    seed: int = 0,
    group_key: str = "source",
    *,
    return_diagnostics: bool = False,
) -> tuple[list[dict], list[dict]] | tuple[list[dict], list[dict], dict]:
    """Make a deterministic grouped split while avoiding a dominant-group overshoot.

    The split prioritizes retaining both labels on both sides when the group structure permits it.
    If there is only one source group it falls back to clip ids, never individual prefixes.
    """
    if not 0.0 < dev_frac < 1.0:
        raise ValueError("dev_frac must be between zero and one")
    if len(rows) < 2:
        raise ValueError("at least two rows are required for a train/dev split")
    rng = random.Random(seed)
    requested_group_key = group_key
    source_groups = {str(row.get(group_key, "unknown")) for row in rows}
    if len(source_groups) < 2:
        group_key = "clip_id"
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[str(r.get(group_key, "unknown"))].append(r)
    if len(by_group) < 2:
        raise ValueError(f"cannot split: only one {group_key!r} group")

    group_names = sorted(by_group)
    rng.shuffle(group_names)
    target = max(1, min(len(rows) - 1, round(dev_frac * len(rows))))
    total_labels = Counter(int(row["label"]) for row in rows)
    group_label_counts = {
        name: Counter(int(row["label"]) for row in by_group[name]) for name in group_names
    }

    def leaves_train_labels(names: set[str]) -> bool:
        selected = sum((group_label_counts[name] for name in names), Counter())
        return all(total_labels[label] - selected[label] > 0 for label in (0, 1))

    # Seed dev with both labels when possible, using small groups so a dominant source cannot
    # consume the entire validation budget on its own.
    selected: set[str] = set()
    mixed = [
        name for name in group_names
        if set(group_label_counts[name]) >= {0, 1} and leaves_train_labels({name})
    ]
    if set(total_labels) >= {0, 1}:
        zero_groups = sorted(
            (name for name in group_names if group_label_counts[name][0] and total_labels[0] > group_label_counts[name][0]),
            key=lambda name: len(by_group[name]),
        )
        one_groups = sorted(
            (name for name in group_names if group_label_counts[name][1] and total_labels[1] > group_label_counts[name][1]),
            key=lambda name: len(by_group[name]),
        )
        for zero_name in zero_groups:
            one_name = next((name for name in one_groups if name != zero_name), None)
            if one_name is not None and leaves_train_labels({zero_name, one_name}):
                selected.update((zero_name, one_name))
                break

    # Prefer two small single-label groups over one dominant mixed source. If that is not
    # feasible, a mixed group still gives both labels without crossing a group boundary.
    if not selected and mixed:
        selected.add(min(mixed, key=lambda name: (abs(len(by_group[name]) - target), len(by_group[name]))))

    if not selected:
        selected.add(min(group_names, key=lambda name: (abs(len(by_group[name]) - target), len(by_group[name]))))

    # Add only groups that improve closeness to the target. This is the key difference from the
    # former "keep adding until >= target" algorithm, which could overshoot by a dominant source.
    while True:
        current = sum(len(by_group[name]) for name in selected)
        candidates = []
        for name in group_names:
            if name in selected:
                continue
            proposed = selected | {name}
            if set(total_labels) >= {0, 1} and not leaves_train_labels(proposed):
                continue
            candidates.append(name)
        if not candidates:
            break
        best = min(candidates, key=lambda name: (abs(current + len(by_group[name]) - target), len(by_group[name])))
        if abs(current + len(by_group[best]) - target) >= abs(current - target):
            break
        selected.add(best)

    train = [row for row in rows if str(row.get(group_key, "unknown")) not in selected]
    dev = [row for row in rows if str(row.get(group_key, "unknown")) in selected]
    achieved_dev_frac = len(dev) / len(rows)
    if (
        group_key == requested_group_key
        and group_key != "clip_id"
        and achieved_dev_frac > max(2.0 * dev_frac, dev_frac + 0.10)
    ):
        # With only a few large sources, strict source grouping can discard an
        # unreasonable fraction of training data. Clip grouping still prevents
        # prefixes of one utterance from leaking across the boundary.
        clip_train, clip_dev, clip_diagnostics = grouped_split(
            rows,
            dev_frac=dev_frac,
            seed=seed,
            group_key="clip_id",
            return_diagnostics=True,
        )
        clip_diagnostics["requested_group_key"] = requested_group_key
        clip_diagnostics["fallback_reason"] = (
            f"source-grouped dev fraction {achieved_dev_frac:.4f} exceeded guardrail"
        )
        if return_diagnostics:
            return clip_train, clip_dev, clip_diagnostics
        return clip_train, clip_dev
    diagnostics = split_diagnostics(
        train, dev, group_key=group_key, requested_group_key=requested_group_key
    )
    if diagnostics["group_overlap"]:
        raise AssertionError("grouped split leaked groups across train and dev")
    if return_diagnostics:
        return train, dev, diagnostics
    return train, dev
