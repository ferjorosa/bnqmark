# Generate Data

Generate the complete BNqMark dataset from scratch: Bayesian networks, queries, and naming variants. This directory contains the data generation pipeline that produces the benchmark datasets. It creates Bayesian networks with controlled structural properties, generates conditional probability queries with ground truth answers computed via exact variable elimination, and produces alternative naming variants for robustness testing.

## Scripts

### `generate_bn_dataset.py`

Generate Bayesian networks with controlled treewidth and distributional properties.

**Key functionality:**
- Generates networks with 4-20 binary variables
- Targets specific treewidths (2, 4, 6, 8, 10, 12) using iterative edge addition
- Creates 2 variants per (n, treewidth) combination with different Dirichlet alpha values
- Alpha=1.0 produces uniform distributions, alpha=0.5 produces skewed/peaked distributions
- Exports to `data/bns.parquet`

**Main function:**
- `generate_bayesian_networks_and_metadata(ns, treewidths, arity_specs, ...)` — Primary entry point

### `generate_query_dataset.py`

Generate conditional probability queries from Bayesian networks.

**Key functionality:**
- Generates queries P(Q|E) with 1-2 target variables and 1-2 evidence variables
- Stratifies by minimum distance between target and evidence (1, 2, or 3 edges)
- Applies informativeness filtering: `|P(Q|E) - P(Q)| >= 0.1`
- Computes ground truth probabilities using exact variable elimination
- Records structural properties (induced width, eliminated variables, distances)
- Exports to `data/queries.parquet`

**Main function:**
- `generate_queries_with_sampling(...)` — Generate queries with threshold filtering

### `generate_naming_variants.py`

Create alternative variable naming strategies for existing Bayesian networks.

**Key functionality:**
- Applies different naming strategies (simple, descriptive) to existing BNs
- Generates renamed network variants while preserving structure and CPTs
- Used for testing LLM robustness to variable naming conventions

**Main function:**
- `create_bn_naming_variant(bn, naming_strategy, ...)` — Create renamed BN variant

### `store_bns_queries_in_db.py`

Insert generated Bayesian networks and queries into the SQLite database.

**Key functionality:**
- Reads `data/bns.parquet` and `data/queries.parquet`
- Inserts into `discrete_bns` and `discrete_queries` tables
- Handles duplicate detection and incremental inserts

**Main functions:**
- `insert_bns(data_dir)` — Insert BN records
- `insert_queries(data_dir)` — Insert query records

### `store_experiments_in_db.py`

Insert experiment results from parquet files into the database.

**Key functionality:**
- Reads intermediate experiment result files
- Inserts into `discrete_experiments` table
- Used for importing external experiment runs or recovery

**Main function:**
- `insert_experiments(data_dir)` — Insert experiment records
