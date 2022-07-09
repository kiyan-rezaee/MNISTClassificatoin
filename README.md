# MNISTClassificatoin

Dataset : digits and labels from sklearn datasets\
images : shape (8,8)\
samples : 1797 (train=1200, test=597)\
Algorithm : SVC and KNN\

Result :

SVC result :
```
              precision    recall  f1-score   support

           0       0.98      0.98      0.98        59
           1       0.97      0.98      0.98        61
           2       1.00      0.97      0.98        60
           3       0.96      0.81      0.88        62
           4       0.98      0.95      0.97        61
           5       0.95      0.98      0.97        59
           6       0.98      0.98      0.98        61
           7       0.94      0.98      0.96        61
           8       0.85      0.96      0.91        55
           9       0.93      0.95      0.94        58

    accuracy                           0.95       597
   macro avg       0.96      0.96      0.95       597
weighted avg       0.96      0.95      0.95       597

```

confusion_matrix :
```
[[58  0  0  0  1  0  0  0  0  0]
 [ 0 60  0  0  0  0  0  0  0  1]
 [ 1  0 58  1  0  0  0  0  0  0]
 [ 0  0  0 50  0  2  0  3  7  0]
 [ 0  0  0  0 58  0  0  0  1  2]
 [ 0  0  0  0  0 58  1  0  0  0]
 [ 0  1  0  0  0  0 60  0  0  0]
 [ 0  0  0  0  0  0  0 60  1  0]
 [ 0  1  0  0  0  0  0  0 53  1]
 [ 0  0  0  1  0  1  0  1  0 55]]

 ```
