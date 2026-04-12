## Queries below threshold

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
