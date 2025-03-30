
# Importing models - DO NOT CHANGE
import sys
import numpy as np
from cvxopt import solvers
import cvxopt.solvers                  # cvxopt for solving the dual optimization problem
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('./pulsar_star_dataset.csv')
df.head()                                                           # reading the dataset

X = df.drop('Class', axis=1)
y = df['Class']                                                                                     # splitting the dataset into features and labels
X = X.to_numpy()
y = y.to_numpy()                                                                                    # converting the dataset into numpy array for ease of use
y[y == 0] = -1                                                                                      # converting the labels to -1 and 1, as per the SVM problem formulation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)           # splitting the dataset into train and test set
mean_train = X_train.mean()                                                                         # standardizing the dataset
std_train = X_train.std()

#########################################    code to be filled part a(i)
X_train = (X_train - mean_train)/std_train                                                 # Fill up this '----------' section
X_test = (X_test - mean_train)/std_train                                                  # Fill up this '----------' section
#########################################    End

class SVM(object):

    def linear_kernel(self, x1, x2):                                                            # defining the kernel functions, using numpy vectorisation to speed up the process
        #########################################    code to be filled a(ii)
        return np.dot(x1,x2)                                                          # Fill up this '-----------' section
        ###############################              End                                        

    def __init__(self, kernel_str='linear', C=1.0, gamma=0.1):                                 # initializing the SVM class
        if kernel_str == 'linear':
            self.kernel = SVM.linear_kernel
        else:
            self.kernel = SVM.linear_kernel
            print('Invalid kernel string, defaulting to linear.')
        self.C = C
        self.gamma = gamma
        self.kernel_str = kernel_str
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        kernel_matrix = np.zeros((num_samples, num_samples))                                                    # creating the kernel matrix
        kernel_matrix = self.kernel(self, X, X.T)

        P = cvxopt.matrix(np.outer(y,y) * kernel_matrix)                                                    # creating the matrices for the dual optimization problem, derivation explained in report
        q = cvxopt.matrix(np.ones(num_samples) * -1)
        A = cvxopt.matrix(y, (1,num_samples)) * 1.
        b = cvxopt.matrix(0) * 1.
        G_upper = np.diag(np.ones(num_samples) * -1)
        G_lower = np.identity(num_samples)
        G = cvxopt.matrix(np.vstack((G_upper, G_lower)))
        h_upper = np.zeros(num_samples)
        h_lower = np.ones(num_samples) * self.C
        h = cvxopt.matrix(np.hstack((h_upper, h_lower)))

        solvers.options['show_progress'] = False                                                            # turning off the progress bar of cvxopt
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)                                                      # running the qp solver of cvxopt to solve the dual optimization problem
        a = np.ravel(solution['x'])                                                                         # get the lagrange multipliers from the solution
        support_vectors = a > 1e-4                                                                          # get the support vectors which have non-zero lagrange multipliers
        ind = np.arange(len(a))[support_vectors]                                                            # get the indices of the support vectors for the kernel matrix
        self.a = a[support_vectors]                                                                         # storing the data of the solution in the svm object
        self.support_vectors = X[support_vectors]
        self.y_support_vectors = y[support_vectors]
        #print("%d support vectors out of %d points" % (len(self.a), num_samples))

        self.b = 0                                                                                          # deriving the bias value by enforcing the constraint for b in the svm optimization problem
        for n in range(len(self.a)):
            self.b += self.y_support_vectors[n]
            self.b -= np.sum(self.a * self.y_support_vectors * kernel_matrix[ind[n],support_vectors])
        self.b /= len(self.a)

        if self.kernel_str == 'linear':                                                                     # deriving the weights for the linear kernel
            self.w = np.zeros(num_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.y_support_vectors[n] * self.support_vectors[n]
        else:
            self.w = None                                                                                   # if the kernel is not linear, then the weights are not defined

    def predict(self, X):
        if self.kernel_str == 'linear':                                                                     # if linear, then the prediction is given by the linear combination of the support vectors
            #########################################    code to be filled a(iii)
            y_predict = np.dot(X,self.w) + self.b
            return np.sign(y_predict)                                                             # Fill up this '----------' section                                                           # Fill up this '----------' section
            ##############################              End
        else:
            y_predict = np.sum(self.a * self.y_support_vectors * self.kernel(self, X, self.support_vectors.T), axis=1)  # if not linear, then the prediction is given by the kernel modification to the standard linear version
            #########################################    code to be filled a(iv)
            y_predict +=self.b                                                        # Fill up this '----------' section
            return np.sign(y_predict)                                                            # Fill up this '----------' section
            ##############################              End

# note that running on the full dataset is very slow (3-4 hours), so uncomment the code below and run this cell if you wish to check the results more quickly or apply grid search, comment it out again before running the full dataset
X_train = X_train[:800]
y_train = y_train[:800]
X_test = X_test[:200]
y_test = y_test[:200]

if __name__ == '__main__':

    """
    ALERT: * * * No changes are allowed in this section  * * *
    """

    input_data_one = sys.argv[1].strip()
    
    """  Call to function that will perform the computation. """
    c_value = float(input_data_one)

    svm_linear = SVM('linear', C=c_value)
    svm_linear.fit(X_train, y_train)
    y_pred_linear = svm_linear.predict(X_test)
    print(accuracy_score(y_test, y_pred_linear))