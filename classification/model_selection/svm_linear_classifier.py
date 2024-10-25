import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)


# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# print(x_train)
# print(x_test)

from sklearn.svm import SVC

classifier = SVC(kernel='linear', random_state=0)

classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
print(accuracy_score(y_test, y_pred))