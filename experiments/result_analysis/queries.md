## Queries below threshold (excluding nulls and -1000s)

```
SELECT
    e.model_name,
    COUNT(*) as num_queries,
    SUM(CASE WHEN ABS(q.probability - e.llm_probability) <= 0.1 THEN 1 ELSE 0 END) as num_within_threshold,
    ROUND(
        100.0 * SUM(CASE WHEN ABS(q.probability - e.llm_probability) <= 0.1 THEN 1 ELSE 0 END) / COUNT(*),
        1
    ) as accuracy_pct
FROM discrete_experiments e
JOIN discrete_queries q
    ON e.query_uuid = q.query_uuid
    AND e.naming_strategy = q.naming_strategy
WHERE e.llm_probability IS NOT NULL
    AND e.llm_probability != -1000
    AND e.naming_strategy = 'simple'
    AND e.experiment_type = 'raw_reasoning'
GROUP BY e.model_name;
```
## Inference cost per model (raw-reasoning and code-generation)
SELECT
    de.model_name,
    SUM(json_extract(de.usage_metadata, '$.upstream_inference_cost')) AS total_cost
FROM discrete_experiments de
WHERE de.experiment_type = 'raw_reasoning'
GROUP BY de.model_name
ORDER BY total_cost DESC;

## Inference time

SELECT
    de.model_name,
    SUM(unixepoch(de.finished_at) - unixepoch(de.started_at)) AS total_seconds
FROM discrete_experiments de
WHERE de.experiment_type = 'raw_reasoning'
GROUP BY de.model_name
ORDER BY total_seconds DESC;
