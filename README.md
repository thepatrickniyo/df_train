# Predictive Modeling for Youth Employment (South Africa)
## Project Overview

This repository contains the complete code, analysis, and findings for the project, "Predictive Modeling of Youth Employment Status in South Africa: Informing AI Career Pathway Interventions."
The primary goal is to leverage advanced machine learning techniques, specifically Deep Neural Networks (DNNs) and class weighting, to build a robust framework that can identify Unemployed Tertiary Graduates (youth aged $\le 35$). This demographic represents the ideal target for mission-critical AI career training and intervention programs designed to transition high-potential, educated individuals into the formal economy.
The success of the project is measured by Recall on this minority target class, prioritizing the identification of those who need assistance over general prediction accuracy.

## Key Findings
Mission Success: The final Keras Functional API Deep Neural Network model achieved a Mission-Critical Recall of $72.15\%$ on the target population (Unemployed Tertiary Graduates).
Architectural Validation: The custom two-path Functional API design proved marginally superior to the Sequential DNN and the Random Forest baseline, demonstrating the value of aligning model architecture with domain-specific features (e.g., education level).
Discriminatory Power: The model exhibits strong discriminatory ability, confirmed by an AUC-ROC of $0.868$.

## Repository Contents
`File/Folder`
## Description
`Patrick_Niyogitare_Summative_Assignment_Model_Training_and_Evaluation.ipynb`
The main Jupyter Notebook containing the end-to-end Machine Learning pipeline: data loading, preprocessing, model architecture definitions (TF-DF, Keras Sequential, Keras Functional API), training with class weighting, and full quantitative/visual evaluation. Start here.
youth_employment_prediction_report.md

The final Scholarly Project Report detailing the problem statement, literature review, methodology, results, discussion, and conclusions.
data/

Contains the raw and processed datasets used for the analysis (e.g., QLFS data subsets).
visualizations/
Contains key exported charts (Confusion Matrix, ROC Curve, Feature Importance) used in the final report and presentation.

## How to Run the Analysis
To reproduce the results and train the models, follow these steps:

### Prerequisites
You need a Python environment with the following libraries:
tensorflow (including Keras and TensorFlow Decision Forests)
```
scikit-learn
pandas, numpy
seaborn, matplotlib
Steps
Clone the Repository:
git clone [https://github.com/PatrickNiyogitare/YouthEmploymentPrediction.git](https://github.com/PatrickNiyogitare/YouthEmploymentPrediction.git)
cd YouthEmploymentPrediction
```

### Install Dependencies: (If using a new environment)
```
pip install tensorflow scikit-learn pandas numpy seaborn matplotlib
```


### Run the Notebook:
Open the main notebook in a Jupyter environment (Lab or Notebook) or Google Colab:
`jupyter notebook Patrick_Niyogitare_Summative_Assignment_Model_Training_and_Evaluation.ipynb`

Execute the cells sequentially to perform data loading, preprocessing, train the three comparison models, apply class weighting, and generate all performance metrics and visualizations.
### Project Assets

Full Report: See the youth_employment_prediction_report.md file for the detailed scholarly write-up.
Video Presentation: 5-10 Minute Video Presentation - A presentation detailing the problem, models, and findings.
Author: Patrick Niyogitare
License: MIT License (or appropriate license)
