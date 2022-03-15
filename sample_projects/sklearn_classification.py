%pip install sklearn

import streamlit as st
from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

st.title("Classification Challenge")

st.write("""
         # Explore different classification algorithms and datasets
         Which one of these algorithms is best for our dataset?
         """)

dataset_name = st.sidebar.selectbox(
    "Select a dataset", 
    ("iris", "digits", "wine", "breast_cancer"))

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    "Select a classifier algorithm",
    ("KNN", "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Digits":
        data = datasets.load_digits()
    elif dataset_name == "Wine Dataset":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y
X, y = get_dataset(dataset_name)
st.write("Shape of the dataset: ", X.shape)
st.write("number of classes", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
       max_depth = st.sidebar.slider("Max Depth", 2, 15)
       n_estimators = st.sidebar.slider("Number of Estimators", 2, 100)
       params["max_depth"] = max_depth
       params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == "SVM":
       clf = SVC(C=params["C"])
    elif clf_name == "KNN":
         clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
       clf = RandomForestClassifier(n_estimators = params["n_estimators"], max_depth = params["max_depth"], random_state = 42)
    return clf

clf = get_classifier(classifier_name, params)

# Classification Algorithms

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier: {classifier_name}")
st.write(f"Accuracy: {acc}")
st.write("Classifier: ", classifier_name)
st.write("Parameters: ", params)

# PLOTS

pca = PCA(n_components=2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

# Show the plot
st.pyplot(fig)