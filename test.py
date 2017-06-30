import csv
from sklearn import svm
from sklearn.svm import OneClassSVM



# Supervised learning model with data in rows, each row is a list

f = open('heart.csv')

csv_f = csv.reader(f)
y = []
X = []
for row in csv_f:
    y.append(row.pop())
    X.append(row)

clf = svm.SVC(gamma=.001, C = 100)
clf.fit(X,y)

print(clf.predict(X[1]))

# plot outlier detection, tells if data
yyy = []
XX = []
for row in csv_f:
    yy = row.pop()
    if int(yy) == 0:
        yyy.append(yy)
        XX.append(row)

clf = svm.OneClassSVM(nu=0.261, gamma=0.05)

clf.fit(XX)

print(clf.predict(XX[100]))