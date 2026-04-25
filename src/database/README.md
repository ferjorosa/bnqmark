# Database Layer

SQLite database management for storing Bayesian networks, queries, experiments, and analysis results.

## Modules

- **`database`** — Core database connection and query utilities.
- **`discrete_bn_db`** — Bayesian network table schema and operations.
- **`discrete_queries_db`** — Query generation results storage.
- **`discrete_experiments_db`** — LLM experiment results storage.
- **`discrete_*_analysis_db`** — Specialized tables for trace analysis results.

## Main Functions

- `query_db(query, params)` — Execute a SQL query and return results as a DataFrame.
- `execute_query(query, params)` — Execute a SQL statement.
- `init_db(db_path)` — Initialize the SQLite database with required tables.
- `insert_bn_batch(rows)` — Batch insert Bayesian networks.
- `insert_query_batch(rows)` — Batch insert queries.
- `insert_experiment_batch(rows)` — Batch insert experiment results.
