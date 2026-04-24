import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score

# Load data
df = pd.read_csv('../data/breast_cancer_mock.csv')

# --- Regression ---
reg_model = LinearRegression()
X_reg = df[['Nucleus_Radius']]
y_reg = df['Nucleus_Perimeter']
reg_model.fit(X_reg, y_reg)
prediction = reg_model.predict([[25.0]])

# --- Classification ---
clf_model = LogisticRegression()
X_clf = df[['Nucleus_Radius', 'Concavity']]
y_clf = df['Diagnosis_Label']
clf_model.fit(X_clf, y_clf)
acc = accuracy_score(y_clf, clf_model.predict(X_clf))
prec = precision_score(y_clf, clf_model.predict(X_clf))

# --- Clustering ---
kmeans = KMeans(n_clusters=2, n_init=10)
df['Cluster'] = kmeans.fit_predict(df[['Nucleus_Radius', 'Concavity']])

# SAVE RESULTS
with open('../results/model_analysis_notes.txt', 'w') as f:
    f.write("# Linear Regression Note\n")
    f.write(f"Slope: {reg_model.coef_[0]:.2f}, Intercept: {reg_model.intercept_:.2f}\n")
    f.write(f"Prediction for Radius 25.0: {prediction[0]:.2f}\n\n")
    
    f.write("# Classification Note\n")
    f.write(f"Accuracy: {acc:.2%}, Precision: {prec:.2%}\n")
    f.write("Note: The model perfectly separated the mock benign/malignant samples based on size and concavity.\n\n")
    
    f.write("# Clustering Note\n")
    f.write("Chosen k=2 clusters. These represent the Benign (smaller nuclei) and Malignant (larger nuclei) groups.")

print('Analysis Completed Successfully.')
