import numpy as np
import seaborn as sns
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('qtagg') # requerido en wayland junto con pyqt6
import matplotlib.pyplot as plt

#sklearn
import sklearn as sk
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder, label_binarize

# Models
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay


iris = load_iris()
iris_features = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X = iris_features.copy() # X mayuscula por ser una MATRIZ de datos

print("<----- Dataset Iris ----->")
print("Características:", X.shape)
print("Clases:", iris.target_names)

for i, name in enumerate(iris.target_names):
    print(f"\t{i}: {name}")

# Identificar los tipos de datos 0, 1, 2 con nombres legibles
label_encoder = LabelEncoder()
encode = label_encoder.fit_transform(iris.target_names)
print("Label Encoder:")
for e in encode:
    print('\tClase', e, ':' , label_encoder.inverse_transform([e]))

# Separar datos de entrenamiento y prueba
y = iris.target # y minuscula por se un vector unidimencional (resultados)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=202, stratify=y)
print("\nTrain y Test:")
print('\tX_train: ', X_train.shape, ' X_test: ', X_test.shape) 
print('\ty_test: ', y_test.shape, ' y_train: ', y_train.shape)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=200,
        C=50,
        solver='lbfgs',
        penalty='l2'
    ),
    "Decision Tree": DecisionTreeClassifier(
        criterion='gini',
        max_depth=3,
        min_samples_split=4,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42
    )
}
print()
results = {}
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
print("\nCross Validation")
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
