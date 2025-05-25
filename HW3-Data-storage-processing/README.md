

## Data Storage and Processing

The project leverages the following tools and approaches for efficient data management:

- **MinIO**: S3-compatible object storage for large-scale data. Deployed locally, via Docker, or on Kubernetes for scalability.
- **Pandas**: Used for data processing in various formats (CSV, Parquet, HDF5). Benchmarking performed to select the optimal format based on save/load times.
- **StreamingDataset**: Dataset converted to streaming format for efficient access during model training.
- **ChromaDB**: Vector database for storing and querying vectorized data representations.
- **Parallel Processing**: Multiprocessing applied to accelerate model inference, significantly reducing execution time.

These tools ensure high performance and scalability for data storage, processing, and access.