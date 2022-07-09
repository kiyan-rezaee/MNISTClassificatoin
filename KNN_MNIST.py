import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

data = datasets.load_digits()
X = data.images.reshape((len(data.images), -1))
Y = data.target

# model 
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier() #default k = 5, you can change parameter k and see if accuracy gets better or not!
knn_model.fit(X[:1200], Y[:1200])

# Evaluation
from sklearn import metrics
p = knn_model.predict(X[1200:])
e = Y[1200:]
print('KNN result : ')
print()
print(metrics.classification_report(e, p))
print('confusion_matrix :')
print()
print(metrics.confusion_matrix(e, p))