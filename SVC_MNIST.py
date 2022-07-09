import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# load dataset
data = datasets.load_digits()

# preprocessing
image_label = list(zip(data.images, data.target))
X = data.images.reshape(len(data.images), -1) #flat
Y = data.target

# Model
from sklearn.svm import SVC

svm_classifier = SVC()

svm_classifier.fit(X[:1200], Y[:1200])

test_setX = X[1200:]
test_setY = Y[1200:]

# # print result
# for i in range(len(test_setX)):
#     print(f"predict : {svm_classifier.predict(test_setX)[i]}, target : {test_setY[i]}")

# total_error = 0
# for i in range(len(test_setX)):
#     if svm_classifier.predict(test_setX)[i] != test_setY[i]:
#         print(f"correct : {test_setY[i]}, predict : {svm_classifier.predict(test_setX)[i]}")
#         total_error += 1
# print(total_error)

# evaluation
from sklearn import metrics

predict = svm_classifier.predict(test_setX)
labels = test_setY
print("SVC result : ")
print()
print(metrics.classification_report(labels, predict))
print('confusion_matrix :')
print()
print(metrics.confusion_matrix(labels, predict))