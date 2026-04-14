# Mejoras en el Modelo de Recuperación (BM25)

Este documento resume las recientes optimizaciones implementadas en el motor de búsqueda (basado en BM25) dentro del módulo `retrieval`, diseñadas específicamente para mejorar la relevancia y el rendimiento al trabajar con un corpus de noticias.

## 1. Ajuste fino de parámetros (Fine-Tuning)
Se han ajustado los valores por defecto de la fórmula BM25 en `bm25.py`:
- **`k1 = 1.5`**: Retarda la saturación de la frecuencia del término. En noticias, la repetición frecuente de palabras clave (ej. "elecciones") es un fuerte indicador de relevancia.
- **`b = 0.8`**: Incrementa la penalización por longitud del documento. Las noticias extremadamente largas a menudo contienen secciones tangenciales o de relleno; este valor prioriza noticias más densas y enfocadas al tema.
- **Herramienta de Tuning**: Se agregó el script `tune_bm25.py` que permite probar rápidamente combinaciones en malla de `k1` y `b` evaluando los `top_k=10` resultados visualmente para calibraciones futuras.

## 2. Ponderación por Campo (Aproximación a BM25F)
El índice original se ha dividido en dos: uno para el **título** (`index_title`) y otro para el **contenido** (`index_content`). 
- Durante la búsqueda, el algoritmo evalúa el documento en ambos índices de manera independiente.
- Los scores se combinan otorgando **mayor peso a los títulos**. La fórmula actual es:
  $$\text{Score Total} = (\text{Score Título} \times 2) + (\text{Score Contenido} \times 1)$$
- *Impacto:* Garantiza que los artículos que mencionan los términos de búsqueda en su titular aparezcan primero en el ranking, mejorando drásticamente la calidad percibida.
- *Nota:* Esto requirió actualizar `IndexBuilder` para construir y almacenar `title_inverted_index.json` y `content_inverted_index.json`.

## 3. Filtrado de Documentos Candidatos (Optimización de Performance)
Para reducir el ruido y evitar cálculos costosos para documentos irrelevantes, ahora se incluye un pre-filtro basado en coincidencias:
- Antes de evaluar el score profundo, se cuentan en cuántos términos de la consulta coincide el documento.
- Si la consulta tiene múltiples palabras, se exige que el documento contenga **al menos 2 términos distintos** de los solicitados (o todos, si la consulta tiene menos de 2).
- *Impacto:* Filtra rápidamente una inmensa cantidad de documentos poco relevantes y acelera el tiempo de búsqueda.

## 4. Boost por frecuencia en la consulta (Query Term Weighting)
Se introdujo una ponderación en caso de que el usuario repita una palabra en su consulta (ej. "impuestos impuestos reformas"):
- El motor ahora calcula las frecuencias de cada término *dentro de la consulta* (`qtf`).
- El tramo de score calculado para cada término se multiplica por sus apariciones en la consulta.
- *Impacto:* Entiende mejor la intención y énfasis del usuario, además de computar el logaritmo IDF una sola vez por término único y multiplicar su puntuación, reduciendo redundancia computacional.

## 5. Registro de Resultados
A lo largo del proceso de optimización, se ha utilizado el archivo `results.txt` para almacenar progresivamente los resultados que iban arrojando las distintas consultas. Este historial nos permitió comparar el impacto directo de cada una de las mejoras añadidas sobre el sistema de ranking final.

---
**Instrucciones de ejecución:**
Para probar y observar los resultados de estos cambios, puedes correr:
```bash
python -m app.retrieval.tune_bm25
```