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

- **Best predictor** varied across models, with [e.g., glucose level / BMI / age] consistently ranking high.
- **AUC scores** ranged from [X] to [Y], with Random Forest / AdaBoost achieving the highest performance.
- **Overall best model** for this dataset: [model name], based on AUC and interpretability.

## Interesting Finding (Extra Credit)

[Add a brief sentence about something non-obvious you discovered, e.g., a weak correlation, a surprising predictor, or a pattern across subgroups.]

## How to Run

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook.

## Author

Chloe Wang  
NYU | Computer and Data Science

## Acknowledgments

Assignment for DS-UA 102. Dataset provided by Professor Pascal Wallisch.
