A repository to review ML concepts as well as libraries.

# Scikit-Learn

Open-source ML library that supports supervised and unsupervised learning, as well as other model and data processing
utilities

## 1 - KNN
KNN is a supervised machine learning classification as well as regression model. It finds "k" closest neigbors to a given
input and makes predictions based on the majority class (for classification) or the average values (for regression).
KNN is a lazy learning algorithm as it does not learn from the dataset, instead it stores the dataset and performs
computations only at the time for prediction.

### Steps for KNN Classification

**1 - Import data from dataset**
```
import pandas as pd
data = pd.read_csv('car.data')
```
**2 - Prepare data for KNN Classification**

KNN works on numeric data
```
import numpy as np
from sklearn.preprocessing import LabelEncoder
X = data[[
    'buying',
    'maint',
    'safety'
    ]].values

Le = LabelEncoder()
for i in range(X.shapre[1]):
    X[:, i] = Le.fit_transform(X[:, i])

label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y = data['class'].map(label_mapping)
y = np.array(y)
```

**3 - KNN Model Training**

```
from sklearn import neighbors
from sklearn.model_selection import train_test_split
```
The neighbors module is used to import models that make predictions based on proximity.
The train_test_split module is used to split data into testing and training data
```
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights= 'uniform')
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

knn.fit(X_train, y_train)

```

**4 - Prediction and accuracy scoring**
```
from sklearn import metrics

predictions = knn.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predictions)
print(predictions, accuracy)

print("Actual value: ", y[10])
print("Predicted value: ", knn.predict(X)[10])
```

## 2 - SVM
Support Vector Machine is a supervised machine learning model used for classification and regression tasks. It finds the best boundary (hyperplane) separating two classes. It is useful for binary classification.
