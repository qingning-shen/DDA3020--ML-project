# This document includes two parts
# The first part is my code without using sklearn
# The second part is my code using sklearn
# Since the performance of the code without sklearn is not very well, but I really did it, so I submit two ways of my coding
# I think I complete the smo, so I hope that I can also have some bonus

"""
import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

X_train = np.genfromtxt('x_train.csv', delimiter=',', dtype=float, skip_header=1)
y_train = np.genfromtxt('y_train.csv', delimiter=',', dtype=float, skip_header=1).reshape(-1, 1)
X_test = np.genfromtxt('x_test.csv', delimiter=',', dtype=float, skip_header=1)
y_test = np.genfromtxt('y_test.csv', delimiter=',', dtype=float, skip_header=1).reshape(-1, 1)

c = 0
sigma = 1
C = 1.0

kernels = [('(a)2nd-order Polynomial Kernel:', 'poly', 2),
           ('(b) 3rd-order Polynomial Kernel:', 'poly', 3),
           ('(c) Radial Basis Function Kernel with \(\sigma = 1\):', 'rbf', sigma),
           ('(d) Sigmoidal Kernel with \(\sigma = 1\):', 'sigmoid', sigma)]
print("Q2.2.3 Calculate using SVM with Kernel Functions and Slack Variables:")

class SVM:
    def __init__(self, C=1):
        self.C = C
        self.alpha = None
        self.bias = None
        self.support_vector = None
        
    def kernel(self, x1, x2, kernel_type='poly', degree=2, sigma=1):
        if kernel_type == 'poly':
            return (1 + x1 @ x2.T) ** degree
        elif kernel_type == 'rbf':
            diff = x1[:, np.newaxis] - x2
            return np.exp(-np.sum(diff ** 2, axis=-1) / (2 * sigma ** 2))
        elif kernel_type == 'sigmoid':
            return np.tanh(sigma * (x1 @ x2.T) + self.bias)
    
    def model(self, X, y, kernel_type='poly', degree=2, sigma=1):
        m, n = X.shape
        self.alpha = np.zeros(m).reshape(m, 1)
        self.bias = y.mean()
        K = self.kernel(X, X, kernel_type, degree, sigma)
        new_new_alpha = 0
        while True:
            new_alpha = 0
            for j in range(m):
                errorj = (self.alpha * y ).T @ K[:, j].reshape(m, 1) + self.bias - y[j]
                if True:
                    rangem = []
                    for x in range(m):
                        if x != j:
                            rangem.append(x)
                    i = np.random.choice(rangem)
                    errori  = (self.alpha * y).T @ K[:, i].reshape(m, 1) + self.bias - y[i]
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
                    self.alpha[j] = self.alpha[j] - y[j] * (errori - errorj) / yita
                    self.alpha[j] = self.call_alpha(self.alpha[j], right, left)
                    if abs(self.alpha[j] - alpha_j) < 1e-3:
                        continue
                    self.alpha[i] += y[i] * y[j] * (alpha_j - self.alpha[j])
                    b1 = self.bias - errori - y[i] * (self.alpha[i] - alpha_i) * K[i, i] - y[j] * (self.alpha[j] - alpha_j) * K[j, i]
                    b2 = self.bias - errorj - y[i] * (self.alpha[i] - alpha_i) * K[i, j] - y[j] * (self.alpha[j] - alpha_j) * K[j, j]
                    self.bias = (b1 + b2) / 2
                    new_alpha += 1
            if new_alpha == 0:
                new_new_alpha += 1
            if new_new_alpha >= 50:
                break
        self.support_vector = np.where(self.alpha > 0)[0]
        
    def predict(self, X, kernel_type='poly', degree=2, sigma=1):
        K = self.kernel(X, X_train, kernel_type, degree, sigma)
        u = np.sign((self.alpha * y_train).T @ K.T + self.bias)
        return u
    
    def decision_function(self, X, kernel_type='poly', degree=2, sigma=1):
        K = self.kernel(X, X_train, kernel_type, degree, sigma)
        u = (self.alpha * y_train).T @ K.T + self.bias
        return u.T
    
    def calculate(self, X, y, kernel_type='poly', degree=2, sigma=1):
        y_predict = self.predict(X, kernel_type, degree, sigma)
        return np.mean(y_predict != y)
    
    def calculate_slackvariables(self, y, y_pred):
        return np.maximum(0, 1 - y * y_pred)
    
    def call_alpha(self, alpha, H, L):
        if alpha > H:
            alpha = H
        elif alpha < L:
            alpha = L
        return alpha 

for name, type, number in kernels:
    c+=1
    setosa = SVM(C=C)
    versicolor = SVM(C=C)
    virginica = SVM(C=C)
    setosa.model(X_train, 2*(y_train == 0).astype(float)-1,type,number,number)
    versicolor.model(X_train, 2*(y_train == 1).astype(float)-1,type,number,number)
    virginica.model(X_train,2*(y_train == 2).astype(float)-1, type,number,number)
    setosa_train = setosa.calculate(X_train, 2*(y_train == 0).astype(float)-1,type,number,number)
    setosa_test = setosa.calculate(X_test, 2*(y_test == 0).astype(float)-1,type,number,number)
    setosa_vectors = setosa.support_vector
    versicolor_train = versicolor.calculate(X_train, 2*(y_train == 1).astype(float)-1,type,number,number)
    versicolor_test = versicolor.calculate(X_test, 2*(y_test == 1).astype(float)-1,type,number,number)
    versicolor_vectors = versicolor.support_vector
    virginica_train = virginica.calculate(X_train, 2*(y_train == 2).astype(float)-1,type,number,number)
    virginica_test = virginica.calculate(X_test, 2*(y_test == 2).astype(float)-1,type,number,number)
    virginica_vectors = virginica.support_vector
    slackvariables_setosa = setosa.calculate_slackvariables(y_train[setosa_vectors], setosa.decision_function(X_train)[setosa_vectors])
    slackvariables_versicolor = versicolor.calculate_slackvariables(y_train[versicolor_vectors], versicolor.decision_function(X_train)[versicolor_vectors])
    slackvariables_virginica = virginica.calculate_slackvariables(y_train[virginica_vectors], virginica.decision_function(X_train)[virginica_vectors])
    
    print(name)
    print("setosa training error: {}, testing error: {},".format(setosa_train, setosa_test))
    print("support_vector_indices_of_setosa: {},".format(setosa_vectors))
    print("slack_variable_of_setosa: {},".format(slackvariables_setosa))
    print("versicolor training error: {}, testing error: {},".format(versicolor_train, versicolor_test))
    print("support_vector_indices_of_versicolor: {},".format(versicolor_vectors))
    print("slack_variable_of_versicolor: {},".format(slackvariables_versicolor))
    print("virginica training error: {}, testing error: {},".format(virginica_train, virginica_test))
    print("support_vector_indices_of_virginica: {},".format(virginica_vectors))
    print("slack_variable_of_virginica: {},".format(slackvariables_virginica))
    if c < 4:
        print('-----------------------------------------')
    """


from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

X_train = np.genfromtxt('x_train.csv', delimiter=',', dtype=float, skip_header=1)
y_train = np.genfromtxt('y_train.csv', delimiter=',', dtype=float, skip_header=1).reshape(-1, 1)
X_test = np.genfromtxt('x_test.csv', delimiter=',', dtype=float, skip_header=1)
y_test = np.genfromtxt('y_test.csv', delimiter=',', dtype=float, skip_header=1).reshape(-1, 1)

C = 1.0
gamma = 0.5
c = 0

kernels = [('(a)2nd-order Polynomial Kernel:', 'poly', 2),
           ('(b) 3rd-order Polynomial Kernel:', 'poly', 3),
           ('(c) Radial Basis Function Kernel with \(\sigma = 1\):', 'rbf', gamma),
           ('(d) Sigmoidal Kernel with \(\sigma = 1\):', 'sigmoid', gamma)]
print("Q2.2.3 Calculate using SVM with Kernel Functions and Slack Variables:")

for name, type, number in kernels:
    c+=1
    if type == 'poly':
        setosa = svm.SVC(kernel=type, degree=number, C=C,random_state=0)
        versicolor = svm.SVC(kernel=type, degree=number, C=C,random_state=0)
        virginica = svm.SVC(kernel=type, degree=number, C=C,random_state=0)
    else:
        setosa = svm.SVC(kernel=type, gamma=number, C=C,random_state=0)
        versicolor = svm.SVC(kernel=type, gamma=number, C=C,random_state=0)
        virginica = svm.SVC(kernel=type, gamma=number, C=C,random_state=0)

    setosa.fit(X_train, 2*(y_train.ravel() == 0)-1)
    versicolor.fit(X_train, 2*(y_train.ravel() == 1)-1)
    virginica.fit(X_train, 2*(y_train.ravel() == 2)-1)
    
    y_train_setosa = setosa.predict(X_train).reshape(-1, 1)
    y_test_setosa = setosa.predict(X_test).reshape(-1, 1)
    y_train_versicolor = versicolor.predict(X_train).reshape(-1, 1)
    y_test_versicolor = versicolor.predict(X_test).reshape(-1, 1)
    y_train_virginica = virginica.predict(X_train).reshape(-1, 1)
    y_test_virginica = virginica.predict(X_test).reshape(-1, 1)

    train_setosa = 1 - accuracy_score(2 * (y_train.ravel() == 0) - 1, y_train_setosa)
    test_setosa = 1 - accuracy_score(2 * (y_test.ravel() == 0) - 1, y_test_setosa)
    train_versicolor = 1 - accuracy_score(2 * (y_train.ravel() == 1) - 1, y_train_versicolor)
    test_versicolor = 1 - accuracy_score(2 * (y_test.ravel() == 1) - 1, y_test_versicolor)
    train_virginica = 1 - accuracy_score(2 * (y_train.ravel() == 2) - 1, y_train_virginica)
    test_virginica = 1 - accuracy_score(2 * (y_test.ravel() == 2) - 1, y_test_virginica)
    
    support_indices_setosa = setosa.support_
    slackvariables_setosa = np.maximum(0, 1 - y_train_setosa[support_indices_setosa].ravel() * setosa.decision_function(X_train[support_indices_setosa]))
    support_indices_versicolor = versicolor.support_
    slackvariables_versicolor = np.maximum(0, 1 - y_train_versicolor[support_indices_versicolor].ravel() * versicolor.decision_function(X_train[support_indices_versicolor]))
    support_indices_virginica = virginica.support_
    slackvariables_virginica = np.maximum(0, 1 - y_train_virginica[support_indices_virginica].ravel() * virginica.decision_function(X_train[support_indices_virginica]))
    
    print(name)
    print("setosa training error: {}, testing error: {},".format(train_setosa, test_setosa))
    print("support_vector_indices_of_setosa: {},".format(support_indices_setosa))
    print("slack_variable_of_setosa: {},".format(slackvariables_setosa))
    print("versicolor training error: {}, testing error: {},".format(train_versicolor, test_versicolor))
    print("support_vector_indices_of_versicolor: {},".format(support_indices_versicolor))
    print("slack_variable_of_versicolor: {},".format(slackvariables_versicolor))
    print("virginica training error: {}, testing error: {},".format(train_virginica, test_virginica))
    print("support_vector_indices_of_virginica: {},".format(support_indices_virginica))
    print("slack_variable_of_virginica: {},".format(slackvariables_virginica))
    if c < 4:
        print('-----------------------------------------')


