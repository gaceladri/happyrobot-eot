from argparse import Namespace

import pytest

from eot.modeling.train import exclude_training_ids, validate_resume_args

TRAIN = [dict(id="a", label=0), dict(id="b", label=0), dict(id="c", label=1)]
DEV = [dict(id="d", label=0), dict(id="e", label=1)]


def test_exclusion_preserves_dev_and_training_order():
    assert exclude_training_ids(TRAIN, DEV, ["a"]) == TRAIN[1:]
    assert len(TRAIN) == 3 and len(DEV) == 2


@pytest.mark.parametrize("ids,error", [(["d"], "development"), (["z"], "unknown"), (["a", "a"], "duplicate"), (["c"], "both|labels")])
def test_invalid_exclusions_fail(ids, error):
    with pytest.raises(ValueError, match=error):
        exclude_training_ids(TRAIN, DEV, ids)


def test_resume_rejects_changed_exclusion_contents():
    with pytest.raises(ValueError, match="exclude_train_sha256"):
        validate_resume_args({"args": {"exclude_train_sha256": "old"}}, Namespace(exclude_train_sha256="new"))
