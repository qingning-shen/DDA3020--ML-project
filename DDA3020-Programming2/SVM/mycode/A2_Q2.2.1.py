# This document includes two parts
# The first part is my code without using sklearn
# The second part is my code using sklearn
# Since the performance of the code without sklearn is not very well, but I really did it, so I submit two ways of my coding
# I think I complete the smo, so I hope that I can also have some bonus

"""
import numpy as np

X_train = np.genfromtxt('x_train.csv', delimiter=',', dtype=None, skip_header=1)
y_train = np.genfromtxt('y_train.csv', delimiter=',', dtype=float, skip_header=1).reshape(-1, 1)
X_test = np.genfromtxt('x_test.csv', delimiter=',', dtype=None, skip_header=1)
y_test = np.genfromtxt('y_test.csv', delimiter=',', dtype=float, skip_header=1).reshape(-1, 1)
for i in range(X_train.shape[1]):
    X_train[:, i] = (X_train[:, i] - X_train[:, i].mean()) / X_train[:, i].std()
for i in range(X_test.shape[1]):
    X_test[:, i] = (X_test[:, i] - X_test[:, i].mean()) / X_test[:, i].std()

class SVM:
    def __init__(self, C=1e5):
        self.C = C
        self.alpha = None
        self.bias = None
        self.support_vector = None
        
    def kernel(self, x1, x2):
        return x1 @ x2.T
    
    def model(self, X, y):
        m, n = X.shape
        self.alpha = np.zeros(m).reshape(m,1)
        self.bias = y.mean()
        K = self.kernel(X, X)
        new_new_alpha = 0
        while True:
            new_alpha = 0
            for j in range(m):
                errorj = (self.alpha * y ).T @ K[:, j].reshape(120,1) +self.bias-y[j]
                # if (y[j] * errorj < -1e-3 and self.alpha[j] < self.C)\
                #     or (y[j] * errorj > 1e-3 and self.alpha[j] > 0):
                if True:
                    rangem = []
                    for x in range(m):
                        if x != j:
                            rangem.append(x)
                    i = np.random.choice(rangem)
                    errori  = (self.alpha * y).T @ K[:, i].reshape(120,1) +self.bias-y[i]
                    alpha_i = self.alpha[i].copy()
                    alpha_j = self.alpha[j].copy()
                    if y[i] != y[j]:
                        left = max(0, self.alpha[j] - self.alpha[i])
                        right = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        left = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        right = min(self.C, self.alpha[i] + self.alpha[j])
                    if left == right:
                        continue
                    yita = 2 * K[i, j] - K[i, i] - K[j, j]
                    if yita >= 0:
                        continue
                    self.alpha[j] = self.alpha[j]-y[j] * (errori - errorj) / yita
                    self.alpha[j] = self.call_alpha(self.alpha[j], right, left)
                    if abs(self.alpha[j] - alpha_j) < 1e-3:
                        continue
                    self.alpha[i] += y[i] * y[j] * (alpha_j - self.alpha[j])
                    b1 = self.bias - errori - y[i] * (self.alpha[i] - alpha_i)\
                    * K[i, i] - y[j] * (self.alpha[j] - alpha_j) * K[j, i]
                    b2 = self.bias - errorj - y[i] * (self.alpha[i] - alpha_i)\
                    * K[i, j] - y[j] * (self.alpha[j] - alpha_j) * K[j, j]
                    self.bias = (b1 + b2) / 2
                    new_alpha += 1
            if new_alpha == 0:
                new_new_alpha+=1
            if new_new_alpha>=50:
                break
        self.support_vector = np.where(self.alpha > 0)[0]
        
    def predict(self, X):
        K = self.kernel(X, X_train)
        u = np.sign((self.alpha * y_train).T @ K.T + self.bias)
        return u
    
    def calculate(self, X, y):
        y_predict = self.predict(X)
        return np.mean(y_predict != y)
    
    def call_alpha(self, alpha, H, L):
        if alpha > H:
            alpha = H
        elif alpha < L:
            alpha = L
        return alpha 

setosa = SVM(C=1)
versicolor = SVM(C=1)
virginica = SVM(C=1)
setosa.model(X_train, 2*(y_train == 0).astype(float)-1)
versicolor.model(X_train, 2*(y_train == 1).astype(float)-1)
virginica.model(X_train,2*(y_train == 2).astype(float)-1)
setosa_train = setosa.calculate(X_train, 2*(y_train == 0).astype(float)-1)
setosa_test = setosa.calculate(X_test, 2*(y_test == 0).astype(float)-1)
setosa_vectors = setosa.support_vector
setosa_w = setosa.alpha[setosa_vectors].T @ X_train[setosa_vectors]
setosa_b = setosa.bias
versicolor_train = versicolor.calculate(X_train, 2*(y_train == 1).astype(float)-1)
versicolor_test = versicolor.calculate(X_test, 2*(y_test == 1).astype(float)-1)
versicolor_vectors = versicolor.support_vector
versicolor_w = versicolor.alpha[versicolor_vectors].T @ X_train[versicolor_vectors]
versicolor_b = versicolor.bias
virginica_train = virginica.calculate(X_train, 2*(y_train == 2).astype(float)-1)
virginica_test = virginica.calculate(X_test, 2*(y_test == 2).astype(float)-1)
virginica_vectors = virginica.support_vector
virginica_w = virginica.alpha[virginica_vectors].T @ X_train[virginica_vectors]
virginica_b = virginica.bias

print("Q2.2.1 Calculation using Standard SVM Model:")
print("setosa training error: {}, testing error: {},".format(setosa_train,setosa_test))
print("w_of_setosa: {}, b_for_setosa: {},".format(setosa_w,setosa_b))
print("support_vector_indices_of_setosa: {},".format(setosa_vectors))
print("versicolor training error: {}, testing error: {},".format(versicolor_train, versicolor_test))
print("w_for_versicolor: {}, b_for_versicolor: {},".format(versicolor_w, versicolor_b))
print("support_vector_indices_of_versicolor: {},".format(versicolor_vectors))
print("virginica training error: {}, testing error: {},".format(virginica_train, virginica_test))
print("w_for_virginica: {}, b_for_virginica: {},".format(virginica_w, virginica_b))
print("support_vector_indices_of_virginica: {},".format(virginica_vectors))
"""

          


from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

X_train = np.genfromtxt('x_train.csv', delimiter=',', dtype=float, skip_header=1)
y_train = np.genfromtxt('y_train.csv', delimiter=',', dtype=float, skip_header=1).reshape(-1,1)
X_test = np.genfromtxt('x_test.csv', delimiter=',', dtype=float, skip_header=1)
y_test = np.genfromtxt('y_test.csv', delimiter=',', dtype=float, skip_header=1).reshape(-1,1)

setosa = svm.SVC(kernel='linear', C=1e5, random_state=0)
versicolor = svm.SVC(kernel='linear', C=1e5,random_state=0)
virginica = svm.SVC(kernel='linear', C=1e5,random_state=0)

setosa.fit(X_train, 2*(y_train.ravel() == 0)-1)
versicolor.fit(X_train, 2*(y_train.ravel() == 1)-1)
virginica.fit(X_train, 2*(y_train.ravel() == 2)-1)

y_train_setosa = setosa.predict(X_train).reshape(-1,1)
y_test_setosa = setosa.predict(X_test).reshape(-1,1)
y_train_versicolor = versicolor.predict(X_train).reshape(-1,1)
y_test_versicolor = versicolor.predict(X_test).reshape(-1,1)
y_train_virginica = virginica.predict(X_train).reshape(-1,1)
y_test_virginica = virginica.predict(X_test).reshape(-1,1)

train_setosa = 1 - accuracy_score(2*(y_train.ravel() == 0)-1, y_train_setosa)
test_setosa = 1 - accuracy_score(2*(y_test.ravel() == 0)-1, y_test_setosa)
train_versicolor = 1 - accuracy_score(2*(y_train.ravel() == 1)-1, y_train_versicolor)
test_versicolor = 1 - accuracy_score(2*(y_test.ravel() == 1)-1, y_test_versicolor)
train_virginica = 1 - accuracy_score(2*(y_train.ravel() == 2)-1, y_train_virginica)
test_virginica = 1 - accuracy_score(2*(y_test.ravel() == 2)-1, y_test_virginica)

w_setosa = setosa.coef_[0]
b_setosa = setosa.intercept_[0]
w_versicolor = versicolor.coef_[0]
b_versicolor = versicolor.intercept_[0]
w_virginica = virginica.coef_[0]
b_virginica = virginica.intercept_[0]
support_indices_setosa = setosa.support_
support_indices_versicolor = versicolor.support_
support_indices_virginica = virginica.support_

print("Q2.2.1 Calculation using Standard SVM Model:")
print("setosa training error: {}, testing error: {},".format(train_setosa, test_setosa))
print("w_of_setosa: {}, b_for_setosa: {},".format(w_setosa, b_setosa))
print("support_vector_indices_of_setosa: {},".format(support_indices_setosa))
print("versicolor training error: {}, testing error: {},".format(train_versicolor, test_versicolor))
print("w_for_versicolor: {}, b_for_versicolor: {},".format(w_versicolor, b_versicolor))
print("support_vector_indices_of_versicolor: {},".format(support_indices_versicolor))
print("virginica training error: {}, testing error: {},".format(train_virginica, test_virginica))
print("w_for_virginica: {}, b_for_virginica: {},".format(w_virginica, b_virginica))
print("support_vector_indices_of_virginica: {},".format(support_indices_virginica))
