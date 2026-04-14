# Diabetes Prediction: Comparison of Classification Models

## Background

Diabetes is a major metabolic disorder affecting over 400 million people worldwide. It is also a leading cause of death, responsible for more than 1.5 million annual deaths. A large proportion of these cases are linked to lifestyle choices, making it critically important to identify reliable predictors of diabetes for both prevention and treatment.

In this project, I built and compared five machine learning models to predict diabetes and identify the most important risk factors using a structured dataset.

## Dataset

The dataset used is clean and carefully curated, with minimal missing data. It contains medical and lifestyle-related features commonly associated with diabetes diagnosis.

## Models Used

I implemented and evaluated the following models, restricting methods to those covered in class:

- Logistic Regression
- Support Vector Machine (SVM)
- Single Decision Tree
- Random Forest
- AdaBoost

For each model, I identified:
- The best predictor of diabetes
- The AUC (Area Under the Curve) score

## Key Questions Answered

1. Which predictor is most important for each model?
2. What is the AUC for each model?
3. Which model performs best overall for this dataset?
4. What additional insight can be drawn from the data beyond the main questions?


## Results Summary

- **Best predictor** varied across models, with **GenerelHealth** consistently ranking high.
- **AUC scores** ranged from 0.598 to 0.820, with Random Forest / AdaBoost achieving the highest performance.
- **Overall best model** for this dataset: **AdaBoosting**, based on AUC and interpretability.

## Interesting Finding

AdaBoosting model is the best model to predict Diabetes Status. It has the highest ROC value (0.822). Random forest, which also uses an ensemble as its core, has a very similar ROC value, proving that ensembles have the ability to learn initially weak learners to strong learners. Logistic regression ROC is also relatively high. 

Zodiac sign is included as a predictor (column 22), and across all 5 models it showed an AUC drop of ~0.000, confirming it has absolutely zero predictive value for diabetes — which is exactly what you'd expect scientifically. This serves as a built-in sanity check: if any model had assigned importance to Zodiac, it would be a red flag for overfitting or data leakage. The fact that all models correctly ignored it validates that the permutation importance method is working properly and the models are learning genuine signal rather than noise.

## How to Run

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook.

## Author

Chloe Wang  
NYU | Computer and Data Science

## Acknowledgments

Assignment for DS-UA 102. Dataset provided by Professor Pascal Wallisch.
