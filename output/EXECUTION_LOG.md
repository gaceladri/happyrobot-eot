# Bitácora de ejecución del pipeline de etiquetado (5 sep 2026)

Complementa a `output/LABELING_PIPELINE_PLAN.md` (la propuesta). Aquí va lo que pasó al ejecutarla:
qué se midió, qué falló, qué se cambió y por qué. Orden cronológico. Se actualiza hasta el documento
final de resultados.

## 1. Datos en disco (todo pinneado por revisión)

| Fuente | Estado | Cifras |
|---|---|---|
| Smart Turn v3.2 train EN (`e564e2ac`) | 16,000 clips, 8,000 por etiqueta, `data/raw/smart-turn-v3.2-eng` | Cobertura por `dataset`: liva_1 47.1 %, chirp3_1 19.6 %, midcentury_1 13.1 %, human_5 5.4 %, rime_2 5.3 %, orpheus_* 7.3 %, chirp3_3_short 1.1 %, human_convcollector_1 0.9 %. **Ninguna fuente supera el 50 %** (umbral de abort del plan), pero liva_1 está al límite y el audio humano etiquetado es ~6 %. Se declara. |
| AppTek Call-Center (`b98967d9`) | 873 llamadas, 14 acentos, ambos canales + diarización manual, 33 GB | 94,679 segmentos manuales. wav 16 kHz mono PCM16 por canal. |
| Krisp Turn-Taking Test v1 (`ea19b274`) | 2,730 clips → `data/raw/krisp/clips` (wav + `clips.jsonl`) | shift 976 / hold 1,754 según card (917/1,813 según conteo del parquet en el plan; se reporta lo que salga del `metadata.json`). **Solo test.** |
| LiveKit eot-bench (harness) | clonado en `third_party/eot-bench`, commit `6594d8b3` (28 ago 2026), venv propio (`numpy<2`) con nuestro paquete instalado `--no-deps` | Trae **predicciones publicadas** para EN de 11 sistemas (Smart Turn v3.2, LiveKit v1/mini, Deepgram Flux, ultraVAD, Gradium, AssemblyAI, Soniox, Cartesia, GPT Realtime 2) sobre el mismo `span_set`. R0 (VAD) y R1 (Smart Turn v3.2) salen de ahí sin reejecutar nada, y `compare-models` pone nuestras runs en la misma tabla. |
| Silero VAD | `v6.2.1`, commit `7e30209a`, sha256 `1a153a22…` verificado en carga (`audio.ensure_silero_vad`) | ONNX, CPU, 512 muestras/32 ms + 64 de contexto, como el wrapper oficial v5+. |

## 2. Detector de pausas: lo que se encontró al medir

Banco de pruebas rápido: 40–64 clips del smoke local de Smart Turn, `telephony_augment` a SNR fija, Jaccard de spans de pausa limpio-vs-ruidoso (métrica b del plan).

1. **El detector legacy colapsa con ruido.** Jaccard mediana **0.00** a 12 y 20 dB (media 0.45), 1.00 a 30 dB. Con umbral relativo al pico (`pico − 35 dB`) el suelo de ruido a 20 dB queda por encima del umbral y no hay ninguna pausa: el clip pierde todos sus HOLD internos en silencio. Es la fisura 6 del plan, ahora con número.
2. **Mi primer ensemble (acuerdo por IoU de spans) tiraba el 57 % de las pausas internas** (39 vs 64 del legacy). Causa: la posterior de Silero decae despacio tras el habla; sus pausas empiezan 100–200 ms tarde y para pausas de 250–400 ms el IoU nunca llega a 0.5. Cambio de criterio: lo que importa para un corte en `inicio + g` es que el VAD esté callado *en el instante del corte y hasta que acabe la pausa*, y que su silencio empezara poco después del inicio energético. La confianza pasa a ser el **retardo** (`lag = inicio_silencio_VAD − inicio_pausa_energía`): high ≤ 100 ms, medium ≤ 232 ms (0.2 s + un chunk: el corte a 0.2 s sigue dentro del acuerdo), low si el VAD oye habla al final de la pausa (energía cayó dentro de una palabra: fricativa, respiración). Retardo medido del VAD: p50 16 ms, p90 126 ms.
3. **Bug propio en el adaptativo:** el umbral era `min(p10 + 9 dB, pico − 35 dB)`; con ruido a 20 dB el segundo término gana y el umbral queda *bajo* el suelo de ruido → cero pausas, igual que el legacy. El límite relativo al pico no tiene sentido si el suelo lo pone el ruido; queda como salvaguarda a `pico − 12 dB`. El habla suave que cae bajo el umbral en clips ruidosos la veta el VAD, no el umbral.
4. **Cuantización de frontera:** el chunk de 32 ms que contiene el fin energético de la pausa ya suele contener el onset del habla siguiente (el VAD lo ve antes que la energía con histéresis). Sin tolerancia, pausas reales de 20–30 s en AppTek salían "low". Se compara el último chunk *totalmente dentro* de la pausa y se tolera un chunk de adelanto del VAD.
5. **Estado tras los arreglos (64 clips smoke):** pausas internas ensemble 59 / legacy 64 / energía sola 265; el ensemble descarta 131 dips de energía donde Silero oye habla (candidatos a corte mid-word que la energía sola habría etiquetado como HOLD). Confianza de las internas conservadas: high 21, medium 38.
6. **Jaccard limpio vs PSTN (40 clips smoke, mediana):**

   | SNR | legacy | energía adaptativa | ensemble |
   |---|---|---|---|
   | 12 dB | 0.00 | 0.53 | 0.24 |
   | 20 dB | 0.00 | 0.81 | 0.70 |
   | 30 dB | 1.00 | 0.98 | 1.00 |

   El umbral a priori del plan (mediana ≥ 0.8 a 20 dB) lo pasa la energía adaptativa y **no** el ensemble (0.70) en esta muestra pequeña: el ensemble es un subconjunto y el ruido mueve pausas entre medium y low. La QA sobre 300 clips (`eot-qa`) es la que cuenta; si el ensemble sigue por debajo de 0.8 se reporta tal cual y se discute recall vs precisión (el ensemble compra precisión de corte, no estabilidad de recall).

Tests que fijan estos comportamientos: `tests/test_labeling.py` (histéresis, veto por VAD, tolerancia de un chunk, legacy pierde pausas a 12 dB que el adaptativo conserva, descarte de low, room tone elimina el silencio digital).

## 3. Oráculo dual-channel de AppTek: primeras cifras

Implementado en `src/eot/apptek.py` con las reglas del plan (§2.3) y umbrales fijados antes de entrenar (`BACKCHANNEL_MAX_S=0.6`, `BACKCHANNEL_MAX_WORDS=2`, `EOT_CUSTOMER_SILENT_S=1.0`, `MAX_HOLD_S=5.0`, `MIN_PAUSE_S=0.2`, `MIN_SPEECH_BEFORE_S=0.3`, `AGENT_MERGE_GAP_S=0.3`, `TECH_WINDOW_S=10`). La diarización manual solo aporta texto (palabras, fillers, filtro técnico); el timing sale del audio de cada canal.

Una llamada (`en_US_General_Agriculture_1586590`, 393 s): 68 pausas → 23 HOLD, 6 EOT, 11 low, 10 técnicas, 9 sin habla previa, 3 largas, 3 solape, 2 colisión, 1 demasiado corta → 55 muestras (37 HOLD, 18 EOT). Retardo de respuesta del agente humano en EOT: mediana 1.42 s (lento; es role-play).

Seis llamadas (3 en-US_General, 3 en-IN): 832 muestras, HOLD:EOT **3.3:1** (el benchmark está ~3:1). Exclusiones: 205 `no_speech_before` (murmullos <0.3 s: "mm", "yes"), 121 `overlap_at_pause`, 68 low, 32 `collision`, 11 `customer_backchannel`, 8 `technical`. Las llamadas en-IN tienen muchísimo más solape (47 por llamada frente a 3–11): el oráculo lo excluye y lo cuenta, no lo etiqueta.

Dos ajustes tras mirar salidas reales: la regex técnica dejó fuera `connection`/`lag`/`test` (vocabulario de negocio en telco), y la actividad del canal agente <150 ms se ignora (clics/respiración disparaban `overlap_at_pause`).

Escala completa (en curso al escribir esto): ~100 muestras por llamada → **~85–90k muestras train (11 acentos) y ~20k held-out (en-IN, en-SG, en-GB_SCT)**. Es más que lo minado de Smart Turn; para R3 y R4 se limitará AppTek al tamaño del set de Smart Turn (muestreo aleatorio con semilla, manteniendo el ratio de etiquetas) para que R3 no sea "casi todo AppTek". Decisión tomada antes de ver ningún resultado de test.

## 4. Decisiones no previstas en el plan (tomadas hoy)

- `min_confidence=medium`: las pausas *low* no entran en train; se cuentan (`stats.dropped`) y alimentan el cubo `low_confidence` de la escucha ciega. Las muestras `final`/`whole` se conservan siempre (su etiqueta viene del clip, no del detector) pero llevan la confianza de la cola.
- El relleno de room tone **no se escribe en disco**: los wav de AppTek se guardan crudos con `noise_fill=1` y `MinedDataset` añade ruido rosa a SNR 15–35 dB en tiempo de carga (train y dev). El slice `raw` de la fisura 2 sale gratis evaluando con `noise_fill=False`.
- `write_samples` y `append_jsonl_record` ya no hacen `fsync` por fila salvo `EOT_DURABLE_WRITES=1`. Los minados van con `--workers N`; ese camino escribe los wav directamente (sin tmp+rename) y no soporta `--resume`.
- `pytest` limitado a `tests/` (`third_party/` traía tests del harness que no compilan en nuestro entorno).

## 5. Incidente de infraestructura

Con tres trabajos multi-proceso escribiendo a la vez (AppTek 8 workers + dos minados) el NVMe local (Toshiba RD500) se degradó a ~200 IOPS con **0.7 s por escritura**, todos los workers en estado D, `dd` directo de 200 MB sin terminar en 45 s, presión de IO al 88 %. Causa: miles de ficheros pequeños con `fsync` + `tmp`+`rename` desde 18 procesos. Al parar los escritores el disco volvió a 1.8 GB/s en <1 min. Solución: sin fsync por fila, escritura directa en el camino paralelo y **los tres trabajos en secuencia** (`scripts/label_all.sh`). No afecta a los resultados, sí al tiempo (≈45 min perdidos).

## 6. Etiquetado completo: cifras finales

| Conjunto | Muestras | HOLD:EOT | Notas |
|---|---|---|---|
| Smart Turn minado, **ensemble** (R2) | 55,504 | 1.62:1 | internas 26,332 · finales 21,172 · whole 8,000. Confianza: high 15,411 / medium 31,274 / low 8,819 (las *low* son finales/whole, cuya etiqueta viene del clip). 32,818 dips de energía descartados por veto del VAD. |
| Smart Turn minado, **legacy** (R2') | 35,221 | 2.01:1 | internas 15,530 · finales 11,691 · whole 8,000. Encuentra menos pausas internas y menos colas de silencio. |
| AppTek oráculo, train (11 acentos) | 82,488 | 1.94:1 | 108k pausas candidatas → 38,331 HOLD + 13,073 EOT etiquetadas (47.5 %); excluidas: no_speech_before 18 %, overlap 13.7 %, low 11.2 %, técnico 2.4 %, too_short 2.2 %, colisión 1.7 %, long_pause 1.2 %. |
| AppTek oráculo, held-out (en-IN, en-SG, en-GB_SCT) | 29,732 | — | nunca se entrena con ellos. |
| R3 = ensemble + AppTek (cap = 55,504, estratificado por etiqueta, semilla 0) | 111,008 | | |
| R4 = AppTek solo (cap 55,504) | 55,504 | 1.94:1 | |

Sorpresa: el ensemble encuentra **más** pausas internas que el legacy (26k vs 15.5k) a pesar de descartar 33k dips: el legacy con umbral relativo al pico se ciega en clips con ruido de fondo, que en Smart Turn son muchos (SNR estimada p10 = 19 dB, p50 = 29 dB).

## 7. QA sobre datos reales (`eval/labeling_qa/report.json`)

**(a) Fronteras contra la segmentación manual de AppTek** (26,881 huecos manuales ≥ 0.3 s entre segmentos de cliente; 16,800 son cambios de turno):

- recall@100 ms 0.67, **recall@200 ms 0.80**; error mediano de las emparejadas **40 ms** (p90 120 ms).
- Umbral a priori (recall@100 ≥ 0.9) **no superado**. Desglose de los 5,392 fallos a 200 ms: 2,452 (45 %) el fin manual del segmento cae *dentro* de una pausa detectada (el anotador rellena; la pausa se encontró antes: no es un fallo del detector); 1,311 (24 %) solo había una pausa *low* (veto del VAD); 1,629 (30 %, 6 % del total) sin pausa. Contando las dos primeras, el detector "ve" el 94 % de las fronteras manuales.
- Por acento: en-SG 0.91, en-AU 0.90 … en-GB 0.67, en-IN 0.67, **en-ZA 0.56**. Los canales de esos acentos son más ruidosos o con más solape; se reportan.

**(b) Robustez PSTN** (300 clips Smart Turn, Jaccard limpio vs telefonía a SNR fija, mediana sobre clips con ≥ 1 pausa limpia). Primera versión del check contaba "nada vs nada" como acuerdo 1.0 y daba al legacy mediana 1.0 mientras encontraba **0.0 pausas por clip** en ruido: corregido (se excluyen esos clips).

| SNR | legacy (124 clips) | energía (298) | ensemble (271) |
|---|---|---|---|
| 12 dB | 0.00 | 0.59 | 0.54 |
| 20 dB | **0.00** | **0.84** | **0.82** |
| 30 dB | 0.89 | 0.98 | 0.92 |

Pausas/clip a 20 dB: legacy 1.29 → 0.00; energía 4.74 → 6.27 (inventa pausas); ensemble 2.46 → 2.76. El ensemble supera el umbral (≥ 0.8) y es el único que no colapsa ni infla. **Decisión: ensemble para R2/R3; legacy solo como R2'.**

**(c) Regla mono-canal vs oráculo dual** (51,404 pausas etiquetadas): si en AppTek etiquetáramos como Smart Turn ("HOLD si el cliente vuelve dentro de H segundos, EOT si no"), con H = 2 s la contaminación de HOLD con EOT reales es 0.9 % y se recupera el 97 % de los EOT, pero el 8.7 % de los "EOT" mono-canal son HOLD según el oráculo (el cliente paró > 2 s y el agente no entró). Con H = ∞ (fin de clip) el 25 % de los HOLD serían EOT reales. El agente humano tarda p50 1.0 s (p10 0.36, p90 2.2) en responder tras un EOT.

**(d) Escucha ciega**: preparada en `eval/labeling_qa/listening_audit/` (175 wav: 50 HOLD internos, 50 EOT, 25 whole, 25 low, 25 medium; clave sellada). **Pendiente de hacer por un humano.** Proxy automático (Silero en los últimos 160 ms antes del corte): habla en el corte en 6 % de los EOT, 34 % de los HOLD internos, 60 % de medium, 80 % de low/whole. El 34 % de HOLD internos es el número a vigilar: parte es el retardo de decaimiento de la posterior (mide los 160 ms previos a un corte que está a 200 ms del inicio de la pausa), parte pueden ser cortes en fricativas. Solo la escucha lo resuelve.

**(e) Distribución**: dos umbrales a priori del plan **fallan** en el minado ensemble: la fuente dominante (liva_1) pasa del 47 % de clips al **59 %** de muestras (produce más pausas por clip), y el 48 % de los clips no aporta ninguna pausa interna (umbral 40 %). Se declara; no se re-muestrea a posteriori.

**(f) Cola Krisp** (400 clips; `last_silence_duration` humano vs nuestra cola): ensemble error absoluto p50 35 ms, 82 % dentro de 100 ms (legacy 76 %, energía 79 %); el legacy no encuentra cola en 102/400.

## 8. Entrenamiento (RTX 3090, 3 épocas, batch 32, lr 5e-5, semilla 0, `torch.compile`)

| Run | Muestras | Dev (grupos de `source` apartados) | AUC dev por época | Tiempo |
|---|---|---|---|---|
| R2 ensemble | 55,504 | 11,663 | 0.848 → 0.858 → **0.878** | 2.7 min |
| R2' legacy | 35,221 | 5,188 | 0.858 → 0.894 → **0.902** | 1.8 min |
| R3 ensemble + AppTek | 111,008 | 16,668 | 0.773 → 0.825 → **0.844** | 12 min |
| R4 AppTek solo | 55,504 | 8,989 (acentos apartados) | 0.799 → 0.794 → **0.792** | 5 min |

Los dev no son comparables entre runs (conjuntos distintos). R4 empeora en dev con las épocas: las etiquetas del oráculo son más difíciles (o más ruidosas) que las minadas y el modelo memoriza acentos de train. El dato que importa es el test externo (EoT Bench, Krisp, held-out AppTek), en curso.

## 9. Evaluación: dos correcciones de camino

- `eot-harness predict --revision …` guarda las runs bajo un span set con sufijo de revisión; el span set publicado no lo lleva. Se comprobó que los 1,250 spans son idénticos (id, índice, inicio, fin, etiqueta) y se fusionaron los directorios para que `compare-models` ponga nuestras runs junto a las 11 publicadas. El script lo hace ya con esa comprobación.
- El bootstrap emparejado aplicaba *nuestra* política congelada al otro modelo. Corregido: cada modelo con su propia política congelada para el mismo punto de operación, sobre el mismo remuestreo de turnos. Los adaptadores en streaming (score_mode=max) no tienen score point y se saltan.

Resultados y lectura: `output/DEFENSE_NOTES.md` §1–§3; tablas generadas: `output/RESULTS_SUMMARY.md`; gráficas del harness: `output/figures/`.

## 10. Estado final

Hecho: todo el pipeline, las 4 runs, las 3 evaluaciones con IC, la QA (a, b, c, e, f) y los documentos. Pendiente y fuera de lo automatizable: la escucha ciega humana (`eval/labeling_qa/listening_audit/`, 175 wav, ~40 min). Sugerido: 3 semillas por run y R4 con `--use-context`.
