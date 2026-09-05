# Plan: pipeline de etiquetado robusto + evaluación en EoT Bench (rev. 0, 5 sep 2026)

Estado: **ejecutado** (ver `EXECUTION_LOG.md` para lo que pasó y `DEFENSE_NOTES.md` para resultados). Acompaña a `output/pdf/*.pdf` y `output/CHANGES_RATIONALE.md`.
Alcance: solo inglés. Español fuera hasta nuevo aviso.

## 0. Resumen en cinco líneas

1. Train: Smart Turn v3.2 (EN) re-cortado a nivel de pausa con un detector de pausas nuevo (ensemble, no solo energía) y metadatos de confianza por muestra.
2. Test: LiveKit EoT Bench EN (harness oficial) como principal y Krisp Turn-Taking Test v1 como segundo test externo. **Krisp nunca entra en train: su licencia lo prohíbe.**
3. Demo: AppTek Call-Center etiquetado con el *oráculo dual-channel* (la etiqueta es lo que hizo el otro humano) y añadido al train en una run separada, pre-registrada, para ver si mueve EoT Bench y Krisp.
4. El etiquetador se valida con números antes de entrenar: acuerdo con fronteras manuales de AppTek, robustez a PSTN, acuerdo single-channel vs dual-channel y una escucha ciega con protocolo.
5. Todas las runs audio-only, misma receta, mismo seed, punto de operación congelado en dev, bootstrap por turno en test.

## 1. Fuentes: qué son de verdad (verificado el 5 sep 2026 con el token)

| Fuente | Revisión | Qué trae | Papel | Restricción |
|---|---|---|---|---|
| `pipecat-ai/smart-turn-data-v3.2-train` | `e564e2ac…` | 270,946 clips; `endpoint_bool`, `language`, `synthetic`, `midfiller`, `endfiller`, `dataset` (fuente); sin transcript ni speaker | Train/dev (minado) | — |
| `livekit/eot-bench-data` config `en` | `ca9d98a9…` | 400 turnos, 1,250 decisiones; `silence_spans`, `words`, `messages`; excluye final <0.2 s y spans >5 s | **Test principal** | Solo `validation` público |
| `Krisp-AI/turn-taking-test-v1` | `ea19b274…` | 2,730 clips; `label` shift 917 / hold 1,813; `last_silence_duration` (mediana 0.69 s, mín 0.056); `speaker_id` ×30; duración 3.7–15.5 s | **Test secundario** | LICENSE: solo benchmarking; **prohibido entrenar** |
| `apptek-com/apptek_callcenter_dialogues` | `b98967d9…` | 873 llamadas, 14 acentos; `test/<acc>/audio/*_channel1.wav` (agent) y `*_channel2.wav` (customer) alineados muestra a muestra; `diarization/<acc>/metadata.jsonl` con `segments{start,end,role,speaker_id,text}` manuales (~5 s) | Demo de etiquetado + train experimental | CC BY-SA 4.0; card: "not intended for training". Se usa como demo declarada |

Hechos medidos en `en_US_General_Agriculture_1586590` (una llamada, 393 s):
- Canales: misma longitud; `mixed == ch1 + ch2` (corr 1.000); crosstalk nulo (RMS del canal ajeno 0.0000–0.0002).
- **Silencio digital**: percentil 50 del frame-dB en ch2 = −92 dB, p5 = −120 dB. La plataforma VoIP hace gating/DTX por hablante.
- 54 segmentos; gaps entre segmentos del mismo hablante: mediana 0.30 s (p90 1.38 s); gaps en cambio de turno: mediana 1.12 s, p10 −0.82 s (solape); 8/54 pares consecutivos se solapan.
- Fillers escritos como `(uh)`, no `#um` como dice el paper; false starts con `~` (`audi~ cab~`). Los primeros segmentos contienen problemas técnicos del role-play ("is your audio on?", "it keeps calibrating").

## 2. Arquitectura del pipeline de etiquetado

```
ingest (provenance)  →  pause_detector (ensemble + confianza)  →  labeler (por fuente)  →  QA numérico + escucha  →  samples.jsonl
```

### 2.1 Ingesta
- Smart Turn: `eot-acquire` existente (streaming, shuffle de shards, cuotas por label, sha256, resume). Añadir al `metadata.json` la **cobertura por `dataset`** y por `synthetic/midfiller/endfiller`, y abortar si una fuente supera el 50 % de los clips.
- AppTek (nuevo `eot-ingest-apptek`): por acento, descargar `diarization/<acc>/metadata.jsonl` + `test/<acc>/audio/*_channel2.wav` (customer) y opcionalmente `_channel1.wav` para verificar. Emparejar por stem. Guardar `speaker_id`, `accent`, `domain`.
- Krisp (nuevo `eot-ingest-krisp`): un parquet (~0.9 GB). Convertir a filas del harness: `cut = duration − last_silence_duration + 0.2`, `label = 1 si shift`, agrupar por `speaker_id`. Excluir filas con `last_silence_duration < 0.2` y reportar n. Guardar también los puntos 0.4/0.6 cuando la cola lo permite para una Pareto parcial.

### 2.2 Detector de pausas (sustituye a `silence_spans` como miner; el actual queda como baseline)
Problema actual (`src/eot/audio.py::silence_spans`): umbral `max(pico − 35 dB, −60 dB)`. Un transitorio fija el pico y manda habla normal a "silencio" (corte mid-word); ruido estacionario a 12 dB SNR sube el suelo y no se detecta ninguna pausa (clip sin HOLD internos). Ambos fallos son silenciosos.

Nuevo detector:
1. **Energía adaptativa al suelo**: `thr = p10(dB) + 9 dB`, acotado a `[−60, pico − 35]`; histéresis onset/offset; `min_speech` 100 ms; fusionar pausas separadas por ráfagas <60 ms.
2. **Silero VAD v6 (ONNX, CPU)** por frames de 32 ms; onset 0.5 / offset 0.35; `min_silence` 0.2 s.
3. **Acuerdo**: una pausa es válida si energía y VAD coinciden con IoU ≥ 0.5. El corte se coloca a `max(offset_energía, offset_vad) + g`, g ∈ grid, para no cortar nunca antes del final real de la palabra.
4. **Metadatos por muestra** en `samples.jsonl`: `snr_est_db`, `detectors_agree` (0/1), `pause_iou`, `onset_margin_ms`, `cut_confidence` ∈ {high, medium, low}. La loss puede ponderar por confianza; la auditoría muestrea la cola baja.
5. Opcional v2: veto por word timestamps (Parakeet TDT 0.6B v3 local): ninguna palabra dentro de la pausa y corte ≥ 80 ms después del último `word_end`. Es la única garantía dura contra el mid-word; añade una dependencia y ~1 h de cómputo. Se decide tras ver los números de QA sin él.

### 2.3 Etiquetado por fuente

**Smart Turn (single-channel, etiqueta a nivel de clip)**, ya implementado en `prefix_mining.py`:
- pausa interna ≥0.2 s con habla después → HOLD en 0.2/0.4/0.6 s, `time_to_onset` conocido;
- pausa final de clip completo → EOT en la misma rejilla;
- clip incompleto → HOLD entero, duración censurada.

**AppTek (dual-channel, oráculo humano)**, nuevo `apptek_labeler.py`. Para cada pausa válida ≥0.2 s del canal customer, dentro de una región donde el customer tiene el turno:

| Qué pasa después de la pausa | Etiqueta | Nota |
|---|---|---|
| El customer retoma antes de que el agente empiece | HOLD | `time_to_onset` real; targets fvad |
| El agente empieza un segmento "de turno" (≥0.6 s o ≥3 palabras) antes de que el customer retome, y el customer no vuelve en 1.0 s desde el inicio del agente | EOT | es lo que hizo el humano |
| El agente emite un backchannel (<0.6 s y ≤2 palabras: "mhm", "okay", "right") y el customer continúa | HOLD, `backchannel=1` | slice propio |
| Colisión (ambos empiezan en <0.3 s) o solape del segmento previo | excluir, `ambiguous=1` | contar |
| Última pausa de la llamada | excluir | no hay "después" |
| Segmentos con `audio|mic|calibrat|can you hear` en el texto o antes del primer saludo del agente | excluir | ruido de role-play |

- Audio de la muestra: últimos ≤8 s del **canal customer** hasta el corte. El agente no está en el audio, igual que en producción (el agente TTS se conoce por texto). `agent_text` = texto del último segmento del agente antes de la pausa; se guarda pero **no se usa** en la run principal (ver fisura 4).
- **Relleno del silencio digital**: todo tramo < −80 dB se rellena con room tone real (ruido de fondo de los clips de Smart Turn o ruido coloreado) a SNR aleatoria 15–35 dB, siempre, antes de `telephony_augment`. Se guarda un slice `raw` sin relleno para detectar el atajo (fisura 2).
- Fillers `(uh)`/`#um` y false starts `~` en el segmento previo → flag `disfluent_before_pause=1`; slice, no etiqueta.

**Krisp**: no se etiqueta. Es test.

### 2.4 QA del etiquetador (antes de entrenar; todo va al documento final)

| Check | Cómo | Umbral propuesto (a priori) |
|---|---|---|
| a. Fronteras | Pausas detectadas vs fronteras manuales de AppTek (inicio/fin de segmento), tolerancia 100 ms | recall ≥ 0.9 de gaps manuales ≥0.3 s; error mediano ≤ 60 ms |
| b. Robustez PSTN | 300 clips Smart Turn: spans en limpio vs `telephony_augment` a 12/20/30 dB; Jaccard por clip | Jaccard mediano ≥ 0.8 a 20 dB; comparar detector viejo vs nuevo |
| c. Single vs dual | En AppTek, etiquetar también con la regla single-channel ("¿retoma el mismo hablante?") y medir desacuerdo con el oráculo | reportar tasa; es la cota del sesgo del etiquetado tipo Smart Turn |
| d. Escucha ciega | 50 HOLD internos, 50 EOT, 25 whole, 25 de `cut_confidence=low`, orden aleatorio, sin ver la etiqueta; `listening_audit.jsonl` | ≥ 95 % de cortes válidos (no mid-word) |
| e. Distribución | ratio HOLD:EOT por fuente (objetivo ~3:1 como el benchmark), histograma de pausas, % clips sin pausa interna, cobertura por `dataset`, `synthetic`, acento | ninguna fuente >50 %; % sin pausa <40 % |

Si (a) o (b) fallan con el detector viejo y pasan con el nuevo, ese es el resultado que justifica el trabajo. Si fallan con ambos, se activa la v2 (word timestamps) antes de entrenar.

## 3. Protocolo experimental pre-registrado

- **Dev**: 15 % de Smart Turn minado, agrupado por `dataset`; prefijos de un clip nunca cruzan. Threshold × action delay × timeout se eligen en dev y se congelan antes de abrir test.
- **Test A**: EoT Bench EN, harness oficial (`livekit/eot-bench` clonado y pinneado). FC @300/@600, delay @5 %/@10 %, detect rate. Bootstrap por turno, 1,000 reps.
- **Test B**: Krisp. Una decisión por clip en 0.2 s; AUC, FC (hold→EOT) vs miss (shift→HOLD) en el punto congelado; Pareto parcial en 0.4/0.6 donde la cola lo permite. Bootstrap por `speaker_id`.
- **Test C (dominio, solo si AppTek entra en train)**: reservar 3 acentos speaker-disjoint como held-out (propuesta: en-IN, en-SG, en-GB_SCT, los peores para ASR en el paper). Mide generalización de acento, no solo memorización.

Runs, todas audio-only, misma receta (whisper-tiny full-FT, lr 5e-5, 3 épocas, fvad on, PSTN 30 %), mismo seed:

| Run | Train | Pregunta |
|---|---|---|
| R0 | — (VAD timer, p=1) | ¿cuánto aporta ML? |
| R1 | pesos públicos Smart Turn v3.2 | ¿mejora la receta publicada? |
| R2 | Smart Turn minado (detector nuevo) | **hipótesis central**: alinear la decisión |
| R2' | Smart Turn minado (detector viejo) | ¿cuánto vale el detector robusto? |
| R3 | R2 + AppTek oráculo (11 acentos) | ¿ayuda el dominio call-center humano-humano? |
| R4 | AppTek solo | control: ¿AppTek basta por sí mismo? |

Hipótesis declaradas: H1 R2 < R1 en FC@300 EN. H2 R2 ≤ R2' (detector). H3 R3 ≤ R2 en Krisp (humano-humano, más cercano a AppTek). H3b en EoT Bench: **incierto por shift de dominio**; se reporta el resultado sea cual sea. Slices obligatorios: clean/PSTN, human/synthetic, midfiller/endfiller, AppTek raw vs relleno, acento.

## 4. Fisuras del plan y mitigaciones

1. **Krisp no puede entrar en train.** La LICENSE lo prohíbe expresamente (training, fine-tuning, derivados). Se usa solo como test. Si la intención era entrenar con él, hay que descartarla.
2. **Silencio digital en AppTek.** −120 dB entre segmentos es un atajo trivial ("cero digital = pausa") que no existe en PSTN. Mitigación: relleno de room tone siempre + `telephony_augment`; slice `raw` para medir el atajo (si el modelo rinde mucho mejor en raw que en relleno, lo está usando). Riesgo residual: el gating puede recortar onsets/offsets de palabra y alterar la prosodia de frontera; se comprueba en la escucha.
3. **Shift de dominio AppTek → EoT Bench.** Humano-humano, role-play, VoIP, cambios de turno lentos (mediana ~1.1 s), solapes frecuentes. EoT Bench es humano-agente con TTS. Entrenar con AppTek puede no mover, o empeorar, EoT Bench. Por eso R3 se pre-registra como hipótesis abierta y se mide también en Krisp (humano-humano). El modelo entregado por defecto es R2; R3 es experimento.
4. **Confound de contexto.** Solo AppTek tendría `agent_text` real; si se activase la rama de contexto, "hay contexto" se correlacionaría con "es AppTek". Todas las runs de este plan son audio-only; el contexto queda para una ablación posterior con datos emparejados.
5. **Definición de EOT en dual-channel.** Backchannels, colisiones, solapes y fin de llamada son ambiguos. Las reglas de §2.3 son explícitas y cada exclusión se cuenta y se reporta. Los umbrales (0.6 s, 3 palabras, 1.0 s, 0.3 s) son propuestas; se fijan antes de entrenar y no se tocan después.
6. **Detector energético actual.** Ver §2.2. Se mantiene como R2' para medir su coste.
7. **Sesgo de muestreo en Smart Turn.** El streaming baraja shards con buffer 2,000; 16k clips EN requieren escanear ~66k filas (~20 shards, ~10 GB). Comprobar cobertura por `dataset` tras la acquisition; si una fuente supera el 50 %, re-muestrear con otra semilla o cuota por fuente.
8. **Krisp: filas con cola <0.2 s** (mín 0.056 s) se excluyen para respetar el score point; ~7 % de clips >8 s se truncan a la ventana, igual que producción. Se reporta n.
9. **Esquema de AppTek ≠ paper.** Fillers `(uh)` en vez de `#um`. Parsear ambos. Primeros segmentos con problemas técnicos: filtrar por regex y por "antes del primer saludo del agente"; declarar el heurístico.
10. **Licencias.** AppTek CC BY-SA 4.0 permite el uso pero la card lo desaconseja para training: demo declarada, no producto. Smart Turn y EoT Bench sin restricción relevante. Krisp: §4.1.
11. **Solapes de fuentes / leakage.** Smart Turn (Pipecat, Liva, Midcentury, MundoAI, sintético), LiveKit (propio), Krisp (Futurebee para Krisp), AppTek (nuevo, no público antes). Independientes por procedencia; no verificable a nivel de audio. Se declara como supuesto.
12. **Dependencia nueva**: Silero VAD v6 vía ONNX (onnxruntime ya está). Pinnear el fichero del modelo por sha. TEN VAD como alternativa si Silero falla el check (a).
13. **Tiempo.** Esto es más de 9 h de ejercicio. Estimación: ingestas 1.5 h (descargas en paralelo), detector + labeler + tests 3 h, QA 1.5 h, runs R2/R2'/R3/R4 en la 3090 <1 h total, harness + Krisp 1.5 h, documento 1.5 h. ≈ 10 h de trabajo efectivo más descargas.

## 5. Decisiones que quedan para el siguiente turno

- AppTek: ¿los 14 acentos (~16 GB solo canal customer + metadata) o 5 acentos para la demo (~6 GB)? Propuesta: 14, reservando 3 como held-out.
- ¿Silero v6 o TEN VAD como segundo detector? Propuesta: Silero (ONNX, sin dependencias nuevas), TEN como fallback.
- ¿Incluir la v2 con word timestamps desde el principio? Propuesta: no; solo si (a)/(b) fallan.
- ¿Seguir usando RunPod? Con la 3090 local no hace falta para estas runs; el runbook queda para reproducibilidad.

## 6. Código previsto

| Fichero | Cambio |
|---|---|
| `src/eot/audio.py` | `silence_spans_adaptive` (histéresis, suelo adaptativo); `SileroVAD` (ONNX); `ensemble_pauses` con IoU y confianza; `fill_digital_silence` |
| `src/eot/prefix_mining.py` | usar el ensemble; emitir metadatos de confianza; flag `--detector legacy|ensemble` |
| `src/eot/apptek.py` (nuevo) | ingest por acento, emparejado de canales, oráculo dual-channel, reglas de §2.3, conteo de exclusiones |
| `src/eot/krisp.py` (nuevo) | ingest, conversión a filas del harness, exclusiones, agrupado por speaker |
| `src/eot/labeling_qa.py` (nuevo) | checks a–e; `listening_audit` CLI; informe JSON |
| `src/eot/train.py` | ponderación opcional por `cut_confidence`; slices por fuente/acento en dev |
| `src/eot/policy.py` | métricas para Krisp (una decisión por clip, Pareto parcial); bootstrap por grupo |
| `scripts/` | clonado pinneado de `livekit/eot-bench`; orquestación local de R0–R4 |
| `tests/` | detector sintético con ruido, reglas del oráculo con segmentos de juguete, conversión Krisp |

Ninguno de estos cambios se ha aplicado. Este documento es la propuesta a validar.
