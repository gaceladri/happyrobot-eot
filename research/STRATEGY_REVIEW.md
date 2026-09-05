# Revisión de estrategia — 2026-09-05

El usuario pide detener la secuencia de pruebas y reconsiderar el camino. D024 se
interrumpió con salida 130: conserva 34 de las 48 llamadas previstas, correspondientes
a 17 casos completos. No se iniciará otro entrenamiento, etiquetado masivo ni prueba
externa como consecuencia de esta revisión. El objetivo SOTA sigue sin cumplirse.

## Lo que establece la evidencia

| Pregunta | Evidencia actual | Interpretación |
|---|---|---|
| ¿Cumplimos el objetivo HTTP? | p95 de 23,6 / 49,0 / 86,7 ms a concurrencia 1 / 4 / 8, 400 solicitudes por nivel, cero errores | Sí en esa máquina y carga; no demuestra toda la latencia de una conversación |
| ¿Nos acercamos a LiveKit v1? | Incumbente 1031,5 ms frente a 543 ms al 5% de cortes; FC300 30,50% frente a 9,93% | La brecha principal es calidad de endpointing; falta reducir aproximadamente un 47% el retraso en ese punto |
| ¿La limpieza de datos funciona? | E013: 979,5 ms, pero falla la réplica E017/E018; E020: 992,5 ms, falla promoción | Señal interesante, todavía insuficiente para atribuir una mejora reproducible al filtrado |
| ¿Más datos bastan? | E009 no mejora con el presupuesto probado | Esa receta falló; no demuestra que cualquier ampliación de datos sea inútil |
| ¿El profesor arregla las etiquetas? | D024 parcial: 34 respuestas válidas, cero revisiones humanas; 5/8 y 6/8 desacuerdos con etiquetas provisionales en los controles aleatorios | El formato funciona; la precisión semántica no está demostrada |
| ¿otoSpeech ha mejorado el modelo? | Audio preparado y verificado; ningún alumno entrenado con ese corpus | No existe todavía evidencia de transferencia |

Fuentes locales: `eval/release_check.json`, `research/baseline_scorecard.json`,
`research/experiments/E013.json`, `E017.json`, `E018.json`, `E020.json`, y
`data/research/D024/report/summary.json`. Los valores de LiveKit provienen de sus
artefactos publicados, no de una nueva ejecución independiente del servicio.

## Dónde falló el enfoque

1. **Buscar ganancias antes de explicar errores.** Se probaron muchas recetas, pero
   seguimos sin una estimación humana de la calidad de las etiquetas ni un desglose
   confirmado de fallos del modelo. Preparar infraestructura, datos y profesores es
   trabajo habilitador; no equivale a mejorar la detección.
2. **Un indicador de desarrollo parcialmente desalineado.** El AUC global decide
   qué candidatos llegan al benchmark; el objetivo real se encuentra en una zona de
   pocos cortes y depende de decisiones temporales. Ese filtro podría descartar
   candidatos útiles o favorecer mejoras irrelevantes. Es una hipótesis pendiente,
   no una explicación demostrada de todos los resultados negativos.
3. **Etiquetas provisionales tratadas como principal señal de selección.** Si el
   desarrollo comparte errores de anotación del entrenamiento, optimizar su AUC no
   identifica necesariamente una mejor intención de fin de turno.
4. **Cambiar de dominio sin demostrar transferencia.** otoSpeech ofrece dos canales
   y eventos conversacionales, pero una conversación entre humanos no tiene siempre
   la misma dinámica que una llamada humano-agente. Fin de segmento, fin de frase,
   cesión efectiva del turno y expectativa de respuesta son objetivos diferentes.
5. **Confundir capacidad general del profesor con capacidad de anotación temporal.**
   En D024 aparecen EOT pese a continuación propia casi inmediata. La cuantización,
   el runtime experimental, la presentación de clips separados y el prompt también
   pueden influir: este piloto no refuta todos los profesores ni el modelo original.
6. **Escala de búsqueda poco proporcionada al tipo de salto necesario.** Una mejora
   aislada de 52 ms no aporta una explicación para cerrar una brecha de 488,5 ms.
   Los ensayos pequeños tampoco descartan que otra representación ayude; sencillamente
   no sabemos aún si falta información, supervisión o capacidad.

La disciplina de controles, réplicas y promociones sí ha sido útil: impidió reemplazar
el incumbente con un resultado frágil. No conviene relajar esas comprobaciones para
obtener una victoria aparente.

## Camino recomendado

**Prioridad inmediata: una auditoría pequeña que decida la siguiente inversión.**
Conservar el modelo servido y pausar la carrera de profesores, corpus y arquitecturas.
Antes de otra tanda, fijar un estudio de aproximadamente 60–100 pausas de entrenamiento
o desarrollo, agrupadas por conversación, con controles aleatorios y casos difíciles.
El muestreo y los criterios se registran antes de ver nuevas respuestas. Esto sirve
para diagnóstico, no para certificar precisión de producción ni superar el gate D019.

La revisión humana tendría dos vistas, juzgadas por separado: el mismo prefijo causal
que ve el detector, y el contexto conversacional completo para adjudicar la etiqueta.
El revisor no ve previamente las propuestas automáticas; los casos disputados requieren
otra revisión. Clasificar: etiqueta/corte incorrectos, información ausente en el prefijo,
evidencia audible que el modelo no utiliza, o ambigüedad genuina. El acuerdo humano
tampoco se presenta como un techo matemático del rendimiento.

| Hallazgo dominante | Siguiente inversión | Prueba que permitiría seguir |
|---|---|---|
| Etiquetas o cortes incorrectos | Corrección estrecha del etiquetador actual | Precisión y cobertura en casos nuevos revisados; una comparación de entrenamiento con control y réplicas |
| El prefijo no contiene información suficiente, el contexto sí | Contexto causal de diálogo o representación temporal | Ganancia en desarrollo con entradas compatibles con la interfaz oficial y latencia total medida |
| Humanos pueden distinguir con el mismo audio pero el modelo falla | Representación preentrenada más capaz, posiblemente destilada | Un único cambio frente a control, dos semillas y mejora en la región de pocos cortes |
| Muchos casos siguen siendo ambiguos | Abstención/incertidumbre y política conservadora | Mejor frontera en desarrollo; cualquier cambio de política se declara y no sustituye el resultado oficial congelado |
| No surge una señal consistente | Consolidar entrega y cerrar búsqueda abierta | Documentación honesta de límites; dejar una próxima hipótesis concreta, no otra tanda automática |

Además propondría un panel de desarrollo que refleje la frontera de pocos cortes y
retardo, manteniendo AUC como auxiliar. Debe declararse prospectivamente y usar datos
de desarrollo independientes; no se cambia el harness, sus grids, las etiquetas del
benchmark ni el objetivo oficial vigente para hacer pasar candidatos anteriores.

**otoSpeech quedaría como segunda fase.** Primero debe mostrar una señal útil en el
piloto y una correspondencia clara entre sus eventos y la expectativa de respuesta.
No etiquetar las 104 horas para descubrir después que las etiquetas no sirven. Un
profesor especializado podría proponer o priorizar casos; no sustituye la validación
independiente. LiveKit requiere permisos específicos para este uso de sus resultados;
la licencia de los pesos EoT de UltraVAD sigue sin aclarar en las fuentes inspeccionadas.

## Ejercicio y objetivo de investigación

El PDF original pide inglés, idealmente menos de 100 ms por solicitud, entrenamiento,
evaluación, API, stress test, Docker, monitorización, documentación y una presentación.
Propone unas nueve horas y desaconseja dedicar demasiado tiempo a optimizar precisión.
No exige superar SOTA. El objetivo ambicioso posterior del usuario es legítimo, pero
conviene distinguirlo de los criterios del ejercicio al decidir dónde invertir tiempo.

Recomendación: priorizar una entrega defendible y esta auditoría explicativa. Mantener
SOTA como meta posterior condicionada a que aparezca una hipótesis con evidencia.
No declarar producción lista: `eval/release_check.json` sigue indicando auditoría humana
pendiente y falta de validación en llamadas representativas.

## Estado conservado

- Incumbente, manifiesto, pesos y ONNX sin cambios; hashes comprobados.
- Los 820 archivos congelados de evaluación/entrenamiento pasan integridad.
- D024 está interrumpido por decisión del usuario; sus resultados son parciales.
  No completar automáticamente el resto ni reinterpretarlo como un piloto terminado.
- Bitácora y registro local actualizados; W&B pendiente por restricción de red.
- No hay nuevo modelo promovido, dato reetiquetado ni evaluación oficial ejecutada.
- Esta propuesta no inicia el estudio humano ni modifica los gates establecidos.

Referencias primarias revisadas:
- Enunciado local: `AI_ML Engineering Task (EoT).pdf`, páginas 2–4.
- https://github.com/livekit/eot-bench
- https://livekit.com/blog/solving-end-of-turn-detection
- https://github.com/livekit/agents/blob/main/MODEL_LICENSE
- https://huggingface.co/fixie-ai/ultraVAD
