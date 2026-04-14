import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

np.random.seed(13798600)

df = pd.read_csv('data/diabetes.csv')
X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13798600, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = AdaBoostClassifier(random_state=13798600)
model.fit(X_train, y_train)
 
proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"Default AUC using all features using AdaBoosting: {auc:.3f}\n")
 
# Permutation importance
importance = {}
for i, col in enumerate(X.columns):
    X_perm = X_test.copy()
    X_perm[:, i] = np.random.permutation(X_perm[:, i])
    drop = auc - roc_auc_score(y_test, model.predict_proba(X_perm)[:, 1])
    importance[col] = drop
    print(f"Removing {col}: AUC drop by {drop:.3f}")
 
importance = pd.Series(importance).sort_values(ascending=False)
 
print(f"\nBest predictor: {importance.idxmax()}")

 
# Plot
colors = ['red' if v == importance.max() else 'steelblue' for v in importance.values[::-1]]
plt.figure(figsize=(8, 6))
plt.barh(importance.index[::-1], importance.values[::-1], color=colors)
plt.title('AdaBoost – Feature Importance (AUC Drop)')
plt.xlabel('AUC Drop')
plt.tight_layout()

 
plt.savefig('graph output: feature importance/random_forest.png', dpi=300, bbox_inches='tight')

plt.show()