# HW6: Testing & CI

This project is part of the "Machine Learning in Production" course, focusing on testing and continuous integration for machine learning applications. It includes the implementation of various testing approaches for code, data, and ML models, as well as model management solutions.

## Project Structure

- `PR1_Code_Testing/`: Unit and integration tests for the ML pipeline codebase.
- `PR2_Data_Testing/`: Tests for data validation, quality assessment, and distribution shifts.
- `PR3_Model_Testing/`: Model evaluation, performance metrics, and robustness testing.
- `PR4_Model_Management/`: Code for storing and versioning models with Weights & Biases.

## Reading List

- [TestPyramid](https://martinfowler.com/bliki/TestPyramid.html)
- [PyTesting the Limits of Machine Learning](https://www.youtube.com/watch?v=GycRK_K0x2s)
- [Testing Machine Learning Systems: Code, Data and Models](https://madewithml.com/courses/mlops/testing/)
- [Beyond Accuracy: Behavioral Testing of NLP models with CheckList](https://github.com/marcotcr/checklist)
- [Robustness Gym](https://github.com/robustness-gym/robustness-gym)
- [ML Testing with Deepchecks](https://github.com/deepchecks/deepchecks)
- [Continuous Machine Learning (CML)](https://github.com/iterative/cml)

## Setup

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Follow the specific setup instructions in each PR folder.

## Running Tests

To run all tests:
