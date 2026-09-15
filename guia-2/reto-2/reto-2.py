import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.datasets import load_iris

from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay


iris = load_iris()
iris_features = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X = iris_features.copy()
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=387, stratify=y) # 328 bayes_accuracy = 1, doble perfect = 356, 387

# Creating a support vector classifier
#model = svm.SVC(probability=True)
param_grid = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "kernel": ["rbf", "poly", "linear", "sigmoid"],
    "gamma": ["scale", "auto", 0.01, 0.1, 1.0, 10.0]
}
#grid = GridSearchCV(model, param_grid)
#grid.fit(X_train, y_train)
#y_pred = grid.predict(X_test)
#
#accuracy = accuracy_score(y_pred, y_test)
#print(f"The model is {accuracy*100}% accurate")
#print(classification_report(y_test, y_pred, target_names=iris.target_names))
#

models = {
    "Naive Bayes": GaussianNB(var_smoothing=2e-9),
    "SVC": GridSearchCV(svm.SVC(probability=True), param_grid)
}

print()
results = {}
flag = False
for name, model in models.items():
    print("<-----", name, "----->")
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    prob = model.predict_proba(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, prediction)
    print("Accuracy:", accuracy)
    results[name] = { "model": model, "pred": prediction, "accuracy": accuracy }

    # Reporte
    print("Classification report:")
    print(classification_report( y_test, prediction, target_names=iris.target_names))

    # Matriz de confusión
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, display_labels=iris.target_names)
    plt.title(f"Matriz de confusión - {name}")
    plt.show()

# Validacion
cv_results = {}
print("\n Cross Validation")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    cv_results[name] = scores
    print(f"{name}: {scores.mean():.4f} std: {scores.std():.4f}")

print("\n<----- Comparacion de modelos ----->")
for name, result in results.items():
    print(f"{name}: Accuracy = {result['accuracy']:.4f}")

print("\nMejor modelo según Accuracy:")
best_model = max( results, key=lambda name: results[name]["accuracy"] )
print( best_model, "con Accuracy =", f"{results[best_model]['accuracy']:.4f}" )

iris_labels = pd.DataFrame( data=iris.target, columns=['target'] )
iris_df = pd.concat( [iris_features, iris_labels], axis=1 )
iris_df['species'] = iris_df['target'].map( lambda x: iris.target_names[x] )
sns.FacetGrid( iris_df, hue='species' ).map( plt.scatter, 'petal length (cm)', 'petal width (cm)' ).add_legend()
plt.title("Dataset Iris") 
plt.show()


"""
- Realizar su implementación del clasificador de flores usando Bayes y máquinas de soporte vectorial.
- Mejorar entendiendo los hiperparámetros y modificándolos para mejorar los resultados
- Validar usando las estrategias vistas.
- Analizar resultados y obtener algunas conclusiones (redactar)
- Guardar evidencia de dicha implementación en un documento. Sustentar los cambios realizados en el modelo. Es importante la fundamentación matemática.
"""
