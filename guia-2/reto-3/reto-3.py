import pandas as pd
import warnings
import matplotlib
matplotlib.use('qtagg') # requerido en wayland junto con pyqt6
import matplotlib.pyplot as plt

#sklearn
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

# Metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

warnings.filterwarnings('ignore')

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
print('\tX_train: ', X_train.shape, ' X_test: ', X_train.shape) 
print('\ty_test: ', y_test.shape, ' y_train: ', y_train.shape)

# inicializacion de lista de inercias
inertias = []     #List
mapping = {}

# rango de numero de centroides
k_range = range(1, 11)

# entrenar y graficar los datos
for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42).fit(X)
    y_kmeans = kmeans.predict(X)

    # guardar las inercias calculadas por kmeans
    inertias.append(kmeans.inertia_)
    mapping[k] = inertias[-1]


print("Inertia values:")
for key, val in mapping.items():
    print(f'{key} : {val}')
# Plotting the graph of k versus Inertia
plt.plot(k_range, inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.grid()
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# graficar los clusters de kmeans vs data set
fig, (ax1, ax2) = plt.subplots(1, 2)

# graficar Kmeans
ax1.scatter(X['petal length (cm)'], X['petal width (cm)'], c=y_kmeans, cmap='viridis', marker='o', edgecolor='k', s=50)
ax1.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s=300, c='red', label='Centroids', edgecolor='k')
ax1.set_title('K-means Clustering (k=3)')
ax1.set_xlabel('petal length')
ax1.set_ylabel('petal width')
ax1.legend()
ax1.grid()

# graficar DataSet
iris_labels = pd.DataFrame( data=iris.target, columns=['target'] )
iris_df = pd.concat( [iris_features, iris_labels], axis=1 )
iris_df['species'] = iris_df['target'].map( lambda x: iris.target_names[x] )
colors = ['#e41a1c', '#377eb8', '#4daf4a']  
for i, species in enumerate(iris.target_names):
    subset = iris_df[iris_df['species'] == species]
    ax2.scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                color=colors[i], marker='o', edgecolor='k', s=50, label=species)
ax2.set_title('Iris DataSet')
ax2.set_xlabel('petal length')
ax2.legend()
plt.show()

# lista de parametros
param_grid = {
    'hidden_layer_sizes': [
        (50,),               # 1 capa oculta con 50 neuronas
        (100,),              # 1 capa oculta con 100 neuronas
        (50, 50),            # 2 capas ocultas con 50 neuronas cada una
        (100, 50, 25),       # 3 capas ocultas decrecientes
        (33, 33, 33)         # 3 capas ocultas de igual tamaño
    ],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter': [25, 50, 100, 200]
}

# busqueda de los mejores hiper parametros con gridsearch
nn = MLPClassifier(random_state=42)
grid_search = GridSearchCV(estimator=nn, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy',)
grid_search.fit(X_train, y_train)
print()
print(f"Mejores hiper parametros: {grid_search.best_params_}")
print(f"Mejor puntuación de precisión (Accuracy): {grid_search.best_score_:.4f}")

nn = MLPClassifier(random_state=42)
nn.fit(X_train,y_train)
models = {
    "Multi Layer Perceptron Default": nn,
    "Multi Layer Perceptron Best HiperParams": grid_search.best_estimator_,
}
print()
results = {}
for name, model in models.items():
    print("<-----", name, "----->")
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
    print(f"{name}: {scores.mean():.4f} +/- {scores.std():.4f}")

print("\n<----- Comparacion de modelos ----->")
for name, result in results.items():
    print(f"{name}: Accuracy = {result['accuracy']:.4f}")

print("\nMejor modelo según Accuracy:")
best_model = max( results, key=lambda name: results[name]["accuracy"] )
print( best_model, "con Accuracy =", f"{results[best_model]['accuracy']:.4f}" )