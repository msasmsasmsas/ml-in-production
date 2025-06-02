
# H6: Testing & CI

## Reading list:

- [TestPyramid](https://martinfowler.com/bliki/TestPyramid.html)
- [PyTesting the Limits of Machine Learning](https://www.youtube.com/watch?v=GycRK_K0x2s)
- [Testing Machine Learning Systems: Code, Data and Models](https://madewithml.com/courses/mlops/testing/)
- [Beyond Accuracy: Behavioral Testing of NLP models with CheckList](https://github.com/marcotcr/checklist)
- [Robustness Gym is an evaluation toolkit for machine learning.](https://github.com/robustness-gym/robustness-gym)
- [ML Testing  with Deepchecks](https://github.com/deepchecks/deepchecks?tab=readme-ov-file)
- [Promptfoo: test your LLM app](https://github.com/promptfoo/promptfoo)
- [The Evaluation Framework for LLMs](https://github.com/confident-ai/deepeval)
- [Continuous Machine Learning (CML)](https://github.com/iterative/cml)
- [Using GitHub Actions for MLOps & Data Science](https://github.blog/2020-06-17-using-github-actions-for-mlops-data-science/)
- [Benefits of Model Serialization in ML](https://appsilon.com/model-serialization-in-machine-learning/)
- [Model registry](https://docs.wandb.ai/guides/model_registry)
- [Privacy Testing for Deep Learning](https://github.com/trailofbits/PrivacyRaven)
- [Learning Interpretability Tool (LIT)](https://github.com/PAIR-code/lit)

## Task:

You need to have a training pipeline for your model for this homework. You can take it from your test task for this course, bring your own or use this [code](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification) as an example.

- Google doc update with a testing plan for your ML model. 
- PR1: Write tests for [code](https://madewithml.com/courses/mlops/testing/#pytest), tests should be runnable from CI.
- PR2: Write tests for [data](https://madewithml.com/courses/mlops/testing/#data), tests should be runnable from CI.
- PR3: Write tests for [model](https://madewithml.com/courses/mlops/testing/#models), tests should be runnable from CI.
- PR4: Write code to store your model in model management with W&B.
- PR5 (optional) : Write code to use [LIT](https://github.com/PAIR-code/lit) for your model, in the case of other domains (CV, audio, tabular) find and use a similar tool.
- PR6 (optional): Write code to test LLM API (select any LLM - OpenAI, VertexAI, etc).

## Criteria:

- 6 PRs merged.
- Testing plan in the google doc.
