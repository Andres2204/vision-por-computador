import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pycaret.classification import setup,compare_models,tune_model,evaluate_model,predict_model,finalize_model,save_model,ensemble_model

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

y = iris.target # y minuscula por se un vector unidimencional (resultados)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print("\nTrain y Test:")
print('\tX_train: ', X_train.shape, ' X_test: ', X_train.shape) 
print('\ty_test: ', y_test.shape, ' y_train: ', y_train.shape)

train_data = X_train
train_data['species'] = y_train
s = setup(data=train_data, target='species', session_id=3)

best = compare_models()

tunned_model = tune_model(best)

print()
print('Bagging')
bagged_model = ensemble_model(tunned_model, method='Bagging')

print()
print('Boosting')
boosted_model = ensemble_model(tunned_model, method='Boosting')

best = bagged_model

evaluate_model(best)

finalize_model(best)

test_data = X_test
test_data['species'] = y_test
predict_model(best, data=test_data)