Design Document for the Project "Organizing Access to a RAG Repository of Information on Threats to Agricultural Crops"
This document outlines the architecture and implementation plan for a project aimed at organizing access to a RAG (Retrieval-Augmented Generation) repository containing instructions, reference materials, and data on threats to agricultural crops (weeds, pests, diseases). The project includes creating a database, a wiki for web and mobile applications, and a RAG repository for processing queries with indexed images. The possibility of fine-tuning a local small-scale model is also considered.

1. Project Objective
Develop a system to provide farmers, agronomists, and other stakeholders with access to structured information about threats to agricultural crops. The system will include:
A database with texts and images about weeds, pests, and diseases.
A wiki interface for web and mobile applications.
A RAG repository for quick search and delivery of relevant information.
The option to fine-tune a local model for query processing.
2. Models in Production
The project will utilize the following models and components:
RAG (Retrieval-Augmented Generation): Used to combine information retrieval from the database with response generation. A pre-trained model (e.g., Llama 3 or BERT) will be adapted for processing queries in Ukrainian.
Local Small-Scale Model: Fine-tuning a model like DistilBERT or MobileBERT for local deployment on resource-constrained servers.
Search Index: Elasticsearch and FAISS for fast text and image retrieval from the database.
Database: PostgreSQL for storing structured information (texts, metadata, categories) and MinIO for storing images.
Architecture
Data Collection: Python scripts (BeautifulSoup, Scrapy) for gathering information from open sources (agricultural websites, scientific articles, databases).
Data Processing: NLP pipeline for text cleaning, categorization, and indexing. Using spaCy and Hugging Face libraries for processing Ukrainian texts.
RAG Repository: Combines FAISS for vector search and Llama 3 for response generation.
API: FastAPI for creating a RESTful API to provide access to the wiki and RAG repository.
Frontend: React for the web application and React Native for the mobile application.





3. Pros and Cons of the Architecture
Pros
Modularity: The architecture allows easy addition of new data sources or models.
Localization: Fine-tuning the model for Ukrainian ensures high-quality responses for local users.
Speed: FAISS and Elasticsearch enable fast searches even with large data volumes.
Accessibility: The wiki interface is adapted for web and mobile platforms, making it convenient for farmers in field conditions.
Cons
Fine-Tuning Complexity: Fine-tuning the model requires a high-quality Ukrainian dataset, which may be challenging.
Local Resource Limitations: A local model may have lower performance compared to cloud-based solutions.
Image Indexing Costs: Processing and indexing large volumes of images can be resource-intensive.

4. Scalability
Horizontal Scaling: Kubernetes for deploying the API and RAG repository allows adding new nodes as load increases.
Search Optimization: FAISS supports distributed indexes for efficient handling of large data volumes.
Caching: Redis for caching frequent queries to reduce database load.
Limitations: Scaling to millions of users may require transitioning to cloud solutions (e.g., AWS or Google Cloud), increasing costs.

5. Usability
Interface: The wiki will feature a simple, intuitive design with keyword search, filters for threat types (weeds, pests, diseases), and image support.
Offline Access: The mobile app will support data caching for use in areas with limited internet connectivity.
Localization: The interface and RAG repository responses will be fully in Ukrainian.
User Training: Documentation and short video tutorials to help farmers quickly adopt the system.

6. Costs
Infrastructure:
Local server: ~$5,000 one-time for hardware (server with GPU for fine-tuning and RAG).
Cloud alternative: ~$200/month for AWS EC2 (g4dn.xlarge) for data processing and API.
Development:
1 developers (backend, frontend, ML) for 6 months: ~$30,000.
Data collection and processing: ~$5,000 (freelancer payments or dataset purchases).
Maintenance:
Technical support and updates: ~$1,000/month.
Cost Optimization: Using open-source models (Llama, DistilBERT) and tools (FAISS, PostgreSQL) reduces dependency on paid services.

7. System Evolution
Short-Term (6 months):
Launch an MVP with a basic wiki and RAG repository.
Index 10,000 texts and 5,000 images.
Medium-Term (1–2 years):
Add support for other languages (English, Polish) for system export.
Integrate with IoT devices for automated threat detection (e.g., via drones).
Long-Term (3+ years):
Implement a full recommendation system for farmers (e.g., pesticide recommendations).
Expand the database for other crop types and regions.

8. Next Steps
Data Collection (1–2 months): Set up scripts for parsing agricultural websites and scientific databases.
Database Development (2 months): Configure PostgreSQL and MinIO, create a schema for texts and images.
Model Fine-Tuning (2–3 months): Prepare a Ukrainian dataset, fine-tune DistilBERT or Llama 3.
API and Frontend Development (3–4 months): Build FastAPI and React/React Native applications.
Testing (1 month): Conduct load testing, verify RAG response quality.
MVP Launch (6th month): Deploy the system on a local server or in the cloud.

9. ML Test Score Evaluation
Evaluation based on the ML Test Score rubric:
Category
Implementation Description
Score (0–4)
Data Tests
Automated data cleaning, consistency checks for texts and images.
3
Model Tests
Testing RAG on a query test set, evaluating response relevance.
3
Infrastructure Tests
Using CI/CD (GitHub Actions), monitoring via Prometheus/Grafana.
4
Production Tests
Load testing API, fault-tolerance checks via Kubernetes.
3
Monitoring
Logging queries and responses, quality metrics (latency, accuracy).
3

Total Score: 16/20. The system meets most criteria but requires additional data and production testing.

10. Business Metrics
Based on the C3.ai example:
Query Processing Time: <2 seconds for 95% of RAG repository queries.
Response Accuracy: >85% relevant responses (measured via user feedback).
Active Users: 1,000 farmers in the first year, growing to 10,000 in 3 years.
Economic Impact: 20% reduction in agronomist consultation costs through automated information access.
Engagement: >70% of users return to the system monthly.

11. Compliance with Recommended Practices
Comparison with recommendations from ml-design-docs and Google Best Practices:
Structure Clarity: The document clearly outlines the objective, architecture, and implementation plan. ✅
Scalability: Kubernetes and FAISS align with recommendations. ✅
Monitoring: Proposed tools (Prometheus, Grafana) meet standards. ✅
Testing: Requires more detailed data testing plans (partial compliance). ⚠️
Business Metrics: Specific metrics included, but ROI analysis needs refinement. ⚠️

12. Evaluation Criteria
Status: Approved with conditions for refinement.
Notes:
Add a detailed data and production testing plan.
Refine ROI evaluation for business metrics.
Repeat the task at the end of the course to compare progress.

13. Conclusion
This design document proposes a realistic plan for creating a RAG repository for information on threats to agricultural crops. The architecture is modular, scalable, and tailored to local users. Next steps include data collection, model fine-tuning, and interface development. The system has the potential for significant economic impact in Ukraine’s agricultural sector.
14. Data Storage and Processing 
Storage Solutions
Our data infrastructure relies on a multi-tiered approach to efficiently handle different types of data and access patterns:
Object Storage with MinIO:


Primary storage solution for large datasets and unstructured data
Deployed on Kubernetes for scalability and reliability
Self-hosted alternative to S3-compatible cloud storage
Used for storing raw datasets, trained models, and intermediate processing results
Vector Database:


Optimized for storing high-dimensional vectors from our embedding models
Enables efficient similarity search and semantic retrieval
Used for quick feature comparison and retrieval during inference
Format Optimization:


Benchmarked performance of various data formats (CSV, Parquet, HDF5, Feather)
Selected Parquet as primary format for tabular data due to superior compression and read performance
Using HDF5 for specific numerical datasets that require partial loading
15. Data Processing Pipeline
Our data processing workflow consists of several stages:
Data Ingestion:


CRUD client for MinIO handles raw data upload and organization
Data validation layer ensures consistency and quality
Metadata tracking for dataset versioning
Data Transformation:


Pandas-based processing for tabular data with optimized memory usage
Streaming processing for larger-than-memory datasets
Parallel processing framework for compute-intensive transformations
Feature Engineering:


StreamingDataset implementation for efficient training data delivery
On-the-fly transformations to reduce storage requirements
Vector embedding generation for semantic search capabilities
Model Serving:


Optimized inference pipeline with benchmarked performance
Multi-process execution for handling concurrent requests
Caching strategy for frequently accessed data
16. Performance Considerations
Our benchmarking experiments revealed several important insights:
Storage Format Impact:


Parquet provides 3.5x faster read times compared to CSV
Compression reduces storage needs by 68% with minimal performance impact
Inference Optimization:


Multi-process approach improved throughput by 2.8x for batch inference
Memory usage optimized through streaming data loading
Scalability:


MinIO on Kubernetes provides horizontal scaling capabilities
Processing pipeline designed to accommodate growing data volumes
This infrastructure ensures efficient data management throughout the ML lifecycle, from initial storage to model deployment, with considerations for performance, scalability, and resource utilization.



## 17. Experiment Management & Model Documentation
### 17.1. Experiment Tracking Infrastructure
To ensure experiment reproducibility, facilitate model comparison, and maintain version control, we have implemented the following infrastructure:
#### 17.1.1. Tools
- **Weights & Biases (W&B)**: For visualization and tracking of training metrics
- **DVC (Data Version Control)**: For versioning models and datasets
- **Git LFS**: For managing large files in the repository
- **Model Card Toolkit**: For standardized documentation of models

#### 17.1.2. Experiment Organization
- Experiments are structured by problem domains (PR1, PR2, PR3)
- Each experiment tracks key metrics including accuracy, loss, F1-score, and resource utilization
- Standardized metadata across all runs facilitates comparison and analysis
- Hyperparameter optimization experiments are systematically organized for comparative analysis

#### 17.1.3. Model Versioning
- Semantic versioning (MAJOR.MINOR.PATCH) is applied to all models
- Each model version is linked to its corresponding dataset version
- Version history and lineage are tracked for complete understanding of model evolution
- Model artifacts are automatically saved and versioned with each experiment

#### 17.1.4. Resource Monitoring
- GPU/CPU utilization tracking during training
- Memory consumption analysis
- Training time optimization based on resource usage patterns
- Cost tracking for cloud-based training resources

### 17.2. Model Cards
All production models include comprehensive model cards that document:
- **Model Details**: Architecture, version, training date, authors
- **Intended Use**: Primary use cases and application scenarios
- **Factors**: Relevant factors that influence model behavior
- **Metrics**: Performance evaluation across different metrics and datasets
- **Training Data**: Data sources, preprocessing steps, and potential biases
- **Ethical Considerations**: Potential risks and mitigations
- **Caveats and Recommendations**: Known limitations and usage guidelines

This documentation ensures transparent communication about model capabilities and limitations to all stakeholders.
### 17.3. Hyperparameter Optimization
Our approach to hyperparameter optimization includes:
- Systematic searches using W&B Sweeps with Bayesian optimization
- Parallel experimentation to efficiently explore the parameter space
- Automated reporting of optimal configurations
- Resource-aware optimization strategies to balance performance and computational cost

### 17.4. Integration with CI/CD
The experiment management infrastructure is integrated with our CI/CD pipeline:
- Automated model testing on each code change
- Performance regression detection
- Model deployment only when quality thresholds are met
- Comprehensive experiment logs accessible to all team members

This experiment management infrastructure provides complete reproducibility for all experiments, enables data-driven decisions for model improvements, and streamlines the transition from experimentation to production.

## 18.Testing Plan
Our comprehensive testing approach encompasses several key areas to ensure the reliability and robustness of the system:

Unit Testing:
- Automated tests for individual components using pytest
- Mocking external dependencies for isolation
- Code coverage target of >80% for critical modules

Integration Testing:
- Testing interactions between components
- API contract verification
- Database integration tests with test fixtures

Data Testing:
- Data validation tests for input formats
- Data drift detection using statistical methods
- Quality checks for text and image datasets

Model Testing:
- Performance metrics evaluation (accuracy, F1-score)
- A/B testing for model changes
- Response quality assessment for the RAG system
- Robustness testing with adversarial examples

Load Testing:
- Simulating concurrent user requests
- Performance benchmarking under varying loads
- Stress testing to identify system limits

Deployment Testing:
- Canary deployments for gradual rollout
- Blue-green deployment strategy for zero-downtime updates
- Rollback procedures validation

Continuous Testing:
- CI/CD pipeline with automated test execution
- Nightly regression tests on larger datasets
- Regular security vulnerability scanning