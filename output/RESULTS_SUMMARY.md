# Resumen de resultados (generado por `eot-report summarize`)

## EoT Bench EN (harness oficial, score point 0.2 s, hold spans 0.2–5 s)

| Modelo | FC @300 ms | FC @600 ms | Delay @5 % FC | Delay @10 % FC | AUC | n turnos |
|---|---:|---:|---:|---:|---:|---:|
| LiveKit Turn Detector v1 | 9.9 % | 4.5 % | 543 ms [469–624] | 295 ms | 0.969 | 400 |
| Deepgram Flux | 12.9 % | 9.9 % | 1151 ms | 548 ms | 0.941 | 400 |
| ultraVAD | 27.7 % | 11.9 % | 899 ms [855–949] | 663 ms | 0.884 | 400 |
| LiveKit Turn Detector v1-mini | 27.8 % | 12.1 % | 1070 ms [1005–1132] | 698 ms | 0.890 | 400 |
| R4 ours: AppTek oracle only | 28.8 % | 13.2 % | 1122 ms [1077–1181] | 746 ms | 0.882 | 400 |
| R3 ours: R2 + AppTek oracle (capped) | 29.9 % | 14.3 % | 1184 ms [1135–1231] | 798 ms | 0.866 | 400 |
| SmartTurn v3.2 | 35.2 % | 14.8 % | 1051 ms [961–1131] | 739 ms | 0.845 | 400 |
| AssemblyAI | 49.4 % | 14.6 % | 1049 ms | 713 ms | 0.896 | 400 |
| R2 ours: Smart Turn mined, ensemble detector | 55.6 % | 15.7 % | 1190 ms [1134–1247] | 825 ms | 0.849 | 400 |
| R2' ours: Smart Turn mined, legacy detector | 55.6 % | 14.8 % | 1175 ms [1141–1208] | 840 ms | 0.852 | 400 |
| Gradium | 55.6 % | 12.6 % | 913 ms | 656 ms | 0.971 | 400 |
| Cartesia Ink 2 | – | – | 1056 ms | 911 ms | 0.955 | 400 |
| OpenAI GPT Realtime 2 | – | – | 1143 ms | 824 ms | 0.858 | 400 |
| Soniox | – | 5.5 % | 647 ms | 512 ms | 0.906 | 400 |

## Krisp Turn-Taking Test v1 (una decisión por clip, bootstrap por speaker)

| Modelo | AUC @0.2 s | FC @300 ms | FC @600 ms | Delay @5 % FC | Delay @10 % FC | n hold / n shift |
|---|---:|---:|---:|---:|---:|---:|
| R0 VAD baseline | 0.500 | 100.0 % | 56.2 % | 1000 ms [1000–1000] | 1000 ms | 1699 / 908 |
| R1 Smart Turn v3.2 (public) | 0.896 | 34.2 % | 5.5 % | 614 ms [559–670] | 456 ms | 1699 / 908 |
| R2 ours: Smart Turn mined, ensemble detector | 0.855 | – | 13.1 % | 780 ms [729–826] | 635 ms | 1699 / 908 |
| R2' ours: Smart Turn mined, legacy detector | 0.849 | – | 10.4 % | 780 ms [732–825] | 610 ms | 1699 / 908 |
| R3 ours: R2 + AppTek oracle (capped) | 0.875 | – | 7.4 % | 673 ms [609–742] | 545 ms | 1699 / 908 |
| R4 ours: AppTek oracle only | 0.904 | 23.7 % | 4.5 % | 581 ms [512–663] | 462 ms | 1699 / 908 |

## AppTek held-out (en-IN, en-SG, en-GB_SCT; corte a 0.2 s; relleno de room tone vs crudo)

| Modelo | AUC relleno | AUC crudo | Δ atajo (crudo − relleno) | FC@0.5 relleno | Detect@0.5 relleno | n |
|---|---:|---:|---:|---:|---:|---:|
| R2 ours: Smart Turn mined, ensemble detector | 0.668 | 0.661 | -0.007 | 17.8 % | 33.8 % | 13086 |
| R2' ours: Smart Turn mined, legacy detector | 0.653 | 0.639 | -0.014 | 12.7 % | 21.9 % | 13086 |
| R3 ours: R2 + AppTek oracle (capped) | 0.719 | 0.685 | -0.033 | 24.0 % | 52.0 % | 13086 |
| R4 ours: AppTek oracle only | 0.701 | 0.709 | +0.009 | 40.6 % | 68.7 % | 13086 |
