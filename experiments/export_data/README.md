# Export Data

Export database tables to Apache Parquet format for the HuggingFace dataset release and external analysis tools. Parquet files provide efficient columnar storage for the benchmark data.

## Scripts

### `export_experiments_to_parquet.py`

Export the `discrete_experiments` table to `data/experiments.parquet`.

**Key functionality:**
- Queries all experiment results from the database
- Exports to parquet with efficient compression
- Used as input for HuggingFace dataset upload

**Main function:**
- `main()` — Query database and export to parquet format

**Output:** `data/experiments.parquet` (large file containing all experiment details including prompts and responses)

## Related Files

The `bns.parquet` and `queries.parquet` files are generated directly by the data generation scripts in `generate_data/` and do not require separate export.
