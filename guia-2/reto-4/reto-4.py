import pandas as pd
import matplotlib.pyplot as plt
import warnings
import matplotlib
matplotlib.use('qtagg') # requerido en wayland junto con pyqt6
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from pycaret.classification import setup,compare_models,tune_model,evaluate_model,predict_model,finalize_model,ensemble_model

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

# inicializacion de pycaret
train_data = X_train
train_data['species'] = y_train
s = setup(data=train_data, target='species', session_id=3)

# se comparan todos los modelos disponibles en la libreria y se selecciona el mejor
best = compare_models()

# compara los hiperparametros de un modelo y selecciona el mejor, pero si es peor que el original devuelve el original
tunned_model = tune_model(best)

# realiza el metodo de ensanble de bagging
print()
print('Bagging')
bagged_model = ensemble_model(tunned_model, method='Bagging')

# realiza el metodo de ensanble de boosting
print()
print('Boosting')
boosted_model = ensemble_model(tunned_model, method='Boosting')

# selecciono el mejor de los 2 metodos de ensanble
best = bagged_model

# muestra una figura del pipeline del modelo
evaluate_model(best)

# entrena el modelo con los datos de test internos
finalize_model(best)

# rendimiento del modelo con datos nuevos
test_data = X_test.copy()
test_data['species'] = y_test
predict_model(best, data=test_data)

# Matriz de confusión
plt.rcdefaults()
prediction = best.predict(X_test)
ConfusionMatrixDisplay.from_predictions(y_test, prediction, display_labels=iris.target_names)
plt.title(f"Matriz de confusión")
plt.show()