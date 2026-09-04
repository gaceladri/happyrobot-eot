# Por qué cambió cada cosa (revisión 2, 4 sep 2026)

Este fichero acompaña a los dos PDFs de `output/pdf/` y al código de `src/eot/`. Registra qué se cambió respecto a la revisión 1, qué evidencia lo motivó y qué se decidió no cambiar. Los hechos de la revisión 1 (números de EoT Bench, TurnBench, model cards) se verificaron contra fuentes primarias y se sostienen; lo que no se sostenía era la estrategia.

## Resumen en una tabla

| # | Cambio | Antes (rev. 1) | Ahora (rev. 2) | Dónde |
|---|---|---|---|---|
| 1 | Receta de entrenamiento | encoder congelado, cabeza 0.25–0.5 época | fine-tune completo, receta pública de Smart Turn | `src/eot/train.py`, `model.py` |
| 2 | Datos | clips enteros de Smart Turn | clips re-cortados a nivel de pausa (prefix mining) | `src/eot/prefix_mining.py` |
| 3 | Contexto | ninguno (audio-only) | última frase del agente, fusionada con gate | `model.py` (`use_context`), `eotbench_adapter.py` |
| 4 | Target | binario EOT/HOLD | + cabezas future-speech a 4 horizontes | `model.py` (`use_fvad`), `prefix_mining.py` |
| 5 | Evaluación externa | EoT Bench "si el adapter está listo" | adapter implementado; EoT Bench en ruta crítica | `eotbench_adapter.py`, `policy.py` |
| 6 | Backbone | `whisper-tiny.en` + run ES separada | `whisper-tiny` multilingüe, un checkpoint para EN y ES | `model.py` |
| 7 | Baseline textual | sobre partial ASR, 45 min | eliminado | agenda |
| 8 | Dominio | un slice más | sección propia; augmentación PSTN en training; challenge numérico | `audio.py`, `challenge.py` |
| 9 | Serving | descrito | implementado y medido (16/34/67 ms p95 a c=1/4/8) | `serve.py`, `load_test.py`, `Dockerfile` |
| 10 | Monitoring | proxies | + experimento natural como fuente de labels; KPIs de negocio | PDF modelo §10 |
| 11 | SOTA omitido | — | Kyutai STT semantic VAD, X2-Turn, FastTurn, SoulX-Duplug, otoSpeech-104h, filas ES | PDF modelo §3.2b, §6, §8 |

## 1. Fine-tune completo en vez de encoder congelado

**Qué decía la rev. 1.** "Entrenamiento garantizado: congelar el encoder y entrenar la cabeza 0.25–0.5 época; unfreeze solo si sobra tiempo."

**Qué se encontró.** El `train.py` público de Smart Turn v3 hace fine-tune completo de `openai/whisper-tiny` (nada congelado), lr 5e-5, 4 épocas, batch 384, warmup 20 %, cosine, weight decay 0.01, sobre 270k clips. Es una receta agresiva, no conservadora.

**Por qué importa.** Comparar una cabeza sobre encoder congelado entrenada media época sobre 15k clips contra un modelo así entrenado no es una ablación: es perder por construcción. Whisper-tiny son 8M de parámetros; el fine-tune completo de 20k clips × 3 épocas cabe en minutos en GPU y ~1 h en Apple Silicon. No había razón de coste para congelar.

**Qué se hizo.** `EOTConfig.freeze_encoder=False` por defecto; los hiperparámetros de `train.py` replican los públicos. `--freeze-encoder` se mantiene como ablación explícita ("¿basta la representación genérica?"), con la respuesta esperada documentada: no.

## 2. Prefix mining: la palanca central

**Qué decía la rev. 1.** Muestrear 12–18k clips y entrenar sobre ellos; el "replay causal" era un experimento pequeño al final.

**Qué se encontró.** Smart Turn v3.2 reporta ~94 % de accuracy en su test de clips y 35.2 % de false cutoffs @300 ms en EoT Bench. La diferencia no es capacidad: el modelo entrena "¿este clip está completo?" y el benchmark (y producción) pregunta "en esta pausa de ≥100 ms, ¿el usuario ha cedido?". Cada turno de EoT Bench tiene de media ~2 pausas HOLD antes de la EOT (850 HOLD : 400 EOT). Los clips de Smart Turn contienen esas mismas pausas internas, pero la receta las ignora.

**Por qué importa.** Es la única intervención que ataca directamente la métrica que se va a reportar, y no necesita datos nuevos. Convierte un problema de "qué dataset falta" en "cómo se cortan los que hay".

**Qué se hizo.** `prefix_mining.py`: cada pausa interna ≥200 ms seguida de habla produce muestras HOLD cortadas a 0.2/0.4/0.6 s dentro de la pausa, con `time_to_onset` real; la pausa final de un clip completo produce EOT en la misma rejilla; un clip incompleto que termina a medias se conserva entero como HOLD con duración censurada. El split (`data.grouped_split`) agrupa por fuente y, si no hay fuentes suficientes, por `clip_id`, para que los prefijos del mismo clip nunca crucen train/dev. En el smoke test, 24 clips sintéticos produjeron 100 muestras (64 HOLD internas, 24 EOT, 12 enteras).

**Limitación declarada.** El detector de silencio para minar es energético y offline; hay que auditar el histograma de pausas y escuchar ~50 cortes. Está previsto poder sustituirlo por Silero.

## 3. Condicionar en la última frase del agente

**Qué decía la rev. 1.** "Texto no es gratis": el baseline textual dependía de un ASR streaming del usuario y se dejaba como late fusion opcional.

**Qué se encontró.** El adapter oficial de LiveKit v1 en eot-bench (`livekit_turn_detector_adapter.py`) envía solo PCM por websocket; no envía `messages`. El README del harness dice que cada adapter recibe `audio`, `messages` (historia previa) y el transcript parcial. El líder del benchmark no usa el contexto que el benchmark permite. UltraVAD sí lo usa, pero como LLM de 0.7–8B en GPU.

**Por qué importa.** Hay una señal de texto que sí es gratis y causal: la del agente. Se conoce antes de que el usuario hable, no cuesta latencia y predice la forma de la respuesta ("¿cuál es tu MC number?" → dígitos con pausas internas; "¿algo más?" → respuesta corta). En el dominio HappyRobot (datos estructurados dictados) es probablemente la señal con más valor por coste.

**Qué se hizo.** `hash_context` hashea uni/bigramas del texto del agente a ids; `EOTModel` los embebe y los fusiona con gate sobre el vector pooled; sin tokenizer ni modelo extra, exportable a ONNX (se cambió `EmbeddingBag` por `Embedding` + media enmascarada porque el exportador rechaza `padding_idx`). Dropout de contexto 0.3 en training para que el modelo siga sirviendo sin él. El adapter usa el último mensaje `assistant` de `messages`; la API lo acepta como `?agent_text=`. Smart Turn no tiene contexto, así que el valor real de esta rama se mide en EoT Bench y en el challenge set, y se declara como hipótesis.

## 4. Cabezas de future-speech

**Qué decía la rev. 1.** "Cabeza auxiliar time-to-next-speech: stretch."

**Por qué se promueve.** Los targets salen gratis del prefix mining (se conoce cuándo vuelve a hablar el usuario tras cada pausa interna). Un target graduado regulariza la decisión binaria (Next-Turn 2026, DualTurn FVAD) y, en inferencia, "P(habla en 2 s) baja" es exactamente el trigger que necesita la anticipación especulativa (Endpoint Anticipation 2026, Flux eager). Coste: una capa lineal.

**Qué se hizo.** Cuatro horizontes (0.24/0.64/1.2/2.0 s), BCE enmascarada para duraciones censuradas, peso 0.5; `--no-fvad` para la ablación. El ONNX exporta `p_fvad` junto a `p_eot`.

## 5. EoT Bench en ruta crítica

**Qué decía la rev. 1.** "6:15–7:15: EoT Bench local si el adapter está listo; si no, comparación con artifacts publicados."

**Por qué se cambia.** Sin el harness, ninguna afirmación sobre false cutoffs es comparable con la tabla pública. El contrato del adapter son 10 líneas (`adapter_id`, `score_point`, `predict_batch`). Dejarlo como opcional era la forma más probable de acabar sin el único número que importa.

**Qué se hizo.** `EOTAdapter` cumple el contrato con backend torch u ONNX; `score_rows` reproduce localmente el esquema de predicciones del harness (`id, span_index, timestamp, silence_dur, p_eot, label`) para cualquier dataset con `silence_spans`; `policy.sweep` barre threshold × action delay × timeout y `operating_points` devuelve FC @300/@600 y delay @5 %/@10 %, más el baseline VAD (p=1 en todos los puntos) y el frente de Pareto. El código se prepara antes del reloj; en la agenda, el harness ocupa 3:15–4:15 (EN) y 5:15–6:00 (ES).

## 6. Un backbone multilingüe

**Qué decía la rev. 1.** `whisper-tiny.en` para EN y `whisper-tiny` para el bonus ES, en runs separadas.

**Por qué se cambia.** Smart Turn v3 usa el multilingüe para todo y no hay evidencia de que `.en` mejore EOT en inglés. Dos backbones duplican training, export, parity y load test. Con el multilingüe, el bonus ES se evalúa con el mismo checkpoint en LiveKit ES (1,236 decisiones; referencias públicas v1 ~16 %, Smart Turn ~39 %) y la run ES separada desaparece de la agenda.

## 7. Eliminar el baseline textual sobre partial ASR

Costaba 45 minutos, dependía de un ASR streaming externo con su propia latencia y respondía a una pregunta ("¿qué añade la semántica?") que el harness ya responde con UltraVAD (27.7 %) y LiveKit v1 (9.9 %). Se sustituye por el experimento "(4) + contexto del agente", que mide semántica *sin* ASR y en la misma run.

## 8. El dominio HappyRobot condiciona el modelo

**Qué decía la rev. 1.** PSTN como "slice principal" de evaluación.

**Por qué se amplía.** HappyRobot opera llamadas de carrier sales / dispatch: PSTN desde la cabina (banda 300–3,400 Hz, G.711, ruido de motor), MC/DOT numbers y load IDs dictados en grupos con pausas de 0.5–1.2 s, rates, teléfonos, fuerte proporción de hablantes no nativos y español, llamadas largas, y coste asimétrico (cortar al carrier cuando da un precio es caro). Cada uno de estos puntos apunta a una decisión de diseño concreta, no a un slice.

**Qué se hizo.** `audio.telephony_augment` (band-limit 8 kHz, mu-law, ganancia, ruido 12–35 dB, packet drops) se aplica al 30 % del training, no solo en eval. `challenge.py` genera turnos de dominio con pausas explícitas, decisiones HOLD esperadas y la pregunta del agente; con `--tts kokoro` produce clips compatibles con `eot-mine`. El PDF del modelo tiene una sección §11 con la tabla rasgo → consecuencia → diseño, y la recomendación de elegir el punto de operación por tipo de pregunta.

## 9. Serving implementado y medido

**Qué decía la rev. 1.** Requisitos correctos (ONNX, warm-up, threads, binario, p50/p95/p99), sin números.

**Qué se hizo.** `serve.py` (FastAPI + onnxruntime, body PCM16/WAV, `p_eot` + `p_fvad` + timings separados + sha del modelo, semáforo con 503 al saturar, 409 si el deadline del caller expiró), `load_test.py` (c=1/4/8, p50/p95/p99 cliente y servidor, cold start aparte) y `Dockerfile` (CPU, sin torch). Medido en Apple M4 con 2 intra-op threads y 8 s de audio: forward 10–14 ms; p95 total 16 / 34 / 67 ms a c=1/4/8; 144 req/s a c=8; cold start 17 ms.

**Un hallazgo del smoke test.** El primer warm-up solo pasaba tensores al session; el primer request real pagó 640 ms construyendo el `WhisperFeatureExtractor`. Ahora el warm-up recorre el camino completo. Es exactamente el tipo de detalle que el ejercicio pide demostrar.

## 10. Monitoring: el experimento natural

Si el agente esperó y el usuario siguió hablando, la pausa era HOLD con certeza. Si el agente habló y el usuario no protestó ni retomó en 2 s, era EOT con alta probabilidad. Solo el EOT seguido de retoma inmediato es ambiguo. Con 1M llamadas/mes son >10M decisiones etiquetadas al mes en el dominio exacto y con audio PSTN real. El sesgo (solo se observan los HOLD que la política actual permitió) se corrige con un pequeño porcentaje de tráfico con timeout más largo. Se añaden KPIs de negocio (re-preguntas, duración, captura correcta de MC/rate/teléfono al primer intento) porque un detector mejor tiene que verse ahí o el cuello de botella está en otra parte.

## 11. SOTA que faltaba

- **Kyutai STT `stt-1b-en_fr`** trae semantic VAD integrado con receta y pesos abiertos (TurnBench 0.773 / 0.059 / 1,007 ms). Es la mejor demostración abierta de "ASR + EOT compartidos".
- **X2-Turn** (arXiv 2608.10878, checkpoint `X2-Turn-4B-0812`): cabeza frame-sync de 6 estados sobre Voxtral-Mini Realtime, supervisión anclada al token ASR a 80 ms, τ como knob.
- **FastTurn** (2604.01897) y **SoulX-Duplug** (2603.14877): misma convergencia hacia estado compartido con el ASR; FastTurn publica además un test set dual-channel.
- **otoSpeech-104h**: train público de TurnBench; único corpus abierto dual-channel con word timings y roles. Pasa a "aux/roadmap" en el reporte de datasets.
- **Filas ES de EoT Bench** y filas aproximadas de OpenAI/AssemblyAI/Cartesia, TurnBench ampliado (Gemini 3.1 Live, ESPnet, WavLM, Kyutai).

Cambia la respuesta a "¿puede hacerlo el transcriptor?": ya no es "Parakeet demuestra que sí", es "hay tres sistemas, dos abiertos, y HappyRobot afina su propio ASR, así que esa es la arquitectura objetivo; el detector Whisper separado es el experimento de 9 h y el fallback portable".

## Qué no se cambió, y por qué

- **La familia de modelo** (audio por turno, Whisper-tiny, activado por VAD). Sigue siendo la única que se puede entrenar, medir y servir en CPU en 9 h. Lo que cambia es cómo se entrena y con qué datos.
- **Los datasets** (Smart Turn train, LiveKit test, Krisp opcional). La rev. 1 los eligió bien; el problema era el uso.
- **La honestidad sobre el SOTA.** 9.9 % (LiveKit v1) no se alcanza en 9 h con datos públicos y se dice así. El objetivo declarado (<25 % FC @300 ms, mejor audio-only pequeño de CPU en el harness) es una hipótesis razonada con intervalo (±3 puntos con 1,250 decisiones), no un resultado.
- **Los tres relojes** y la métrica Pareto false-cutoff vs delay. Correctos en la rev. 1; ahora tienen código (`policy.py`).

## Cómo se verificó el código

Smoke test end-to-end ejecutado desde fuera del repositorio sobre 24 clips sintéticos: `eot-mine` (100 muestras, targets y máscaras correctas) → `eot-train` 1 época CPU con contexto y fvad (AUC dev 0.93 sobre 19 muestras; solo prueba que el pipeline funciona) → `eot-export` (parity FP32 4e-7; INT8 aceptado con Δp 0.01) → adapter + `score_rows` + `eot-sweep` (116 filas, esquema del harness, puntos de operación) → `eot-serve` + `eot-loadtest` (0 errores, números arriba) → `eot-challenge` (20 turnos, 39 decisiones HOLD). Lo que no se ha ejecutado, por tamaño: la descarga de Smart Turn (41 GB) y el harness de eot-bench real; ambos están en la agenda "antes del reloj".
