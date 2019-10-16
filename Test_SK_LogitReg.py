import scipy as sp
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

x = np.loadtxt('wine.data', delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13))
y = np.loadtxt('wine.data', delimiter=',', usecols=(0))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)
print(model)

R = y_test
P = model.predict(x_test)

print(metrics.classification_report(R, P))  // precison&recall
print(metrics.confusion_matrix(R, P)) // confusion matrix
