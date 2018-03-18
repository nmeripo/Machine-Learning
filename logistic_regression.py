import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit


class LogisticRegression(object):
    def sigmoid(self, z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """

        s = 1 / (1 + np.exp(np.clip(-z, a_min=None, a_max = np.log(np.finfo(np.float).max))))
        return s


    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)

        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """

        w = np.zeros((dim, 1))
        b = 0

        assert (w.shape == (dim, 1))
        assert (isinstance(b, float) or isinstance(b, int))

        return w, b


    def propagate(self, w, b, X, Y):
        """
        Arguments:
        w -- weights, a numpy array of size (training example size, 1)
        b -- bias, a scalar
        X -- data of size (training example size, number of examples)
        Y -- true "label" vector (containing 1 if the client subscribed a term deposit, else 0) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        """

        m = X.shape[1]

        # FORWARD PROPAGATION (FROM X TO COST)
        A = self.sigmoid(np.dot(w.T, X) + b)  # compute activation
        cost = (-1.0 / m) * np.sum(Y * np.log(A + + 1e-9) + (1 - Y) * np.log(1 - A + 1e-9))  # compute cost

        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = (1.0 / m) * np.dot(X, (A - Y).T)
        db = (1.0 / m) * np.sum(A - Y)

        assert (dw.shape == w.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        grads = {"dw": dw,
                 "db": db}

        return grads, cost


    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (training example size, 1)
        b -- bias, a scalar
        X -- data of shape (training example size, number of examples)
        Y -- true "label" vector (containing 1 if the client subscribed a term deposit, else 0), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

        """

        costs = []

        for i in range(num_iterations):

            # Cost and gradient calculation
            grads, cost = self.propagate(w, b, X, Y)

            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]

            # update rule
            w -= learning_rate * dw
            b -= learning_rate * db

            # Record the costs
            if i % 100 == 0:
                costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w,
                  "b": b}

        grads = {"dw": dw,
                 "db": db}

        return params, grads, costs


    def predict(self, w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (training example size, 1)
        b -- bias, a scalar
        X -- data of size (training example size, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''

        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        A = self.sigmoid(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            Y_prediction[0, i] = 1 if (A[0, i] > 0.5) else 0

        assert (Y_prediction.shape == (1, m))

        return Y_prediction


    def model(self, X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.1, print_cost=True):
        """
        Builds the logistic regression model by calling the function you've implemented previously

        Arguments:
        X_train -- training set represented by a numpy array of shape (training example size, m_train)
        Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
        X_test -- test set represented by a numpy array of shape (training example size, m_test)
        Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
        num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
        learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
        print_cost -- Set to true to print the cost every 100 iterations

        Returns:
        d -- dictionary containing information about the model.
        """

        # initialize parameters with zeros
        w, b = self.initialize_with_zeros(X_train.shape[0])

        # Gradient descent
        parameters, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)

        # Retrieve parameters w and b from dictionary "parameters"
        w = parameters["w"]
        b = parameters["b"]

        # Predict test/train set examples
        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)


        # Print train/test Errors
        print("Train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("Test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train": Y_prediction_train,
             "w": w,
             "b": b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

        return d

    def plot_learning_curve(self, d):
        costs = np.squeeze(d['costs'])
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(d["learning_rate"]))
        plt.show()




if __name__ == "__main__":
    """
    Dataset:
    [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. 
    A Data-Driven Approach to Predict the Success of Bank Telemarketing. 
    Decision Support Systems, Elsevier, 62:22-31, June 2014
    """
    bank_df = pd.read_csv("bank-full.csv", delimiter=";")
    print("Percentage of each class:\n", bank_df['y'].value_counts(normalize=True), "\n")
    bank_df['y'] = pd.factorize(bank_df['y'])[0]

    # Separate majority and minority classes
    bank_df_majority = bank_df[bank_df.y == 0]
    bank_df_minority = bank_df[bank_df.y == 1]

    # Downsample majority class
    bank_df_majority_downsampled = resample(bank_df_majority,
                                       replace=False,  # sample without replacement
                                       n_samples=bank_df_minority.shape[0],  # to match minority class
                                       random_state=123)  # reproducible results

    # Combine minority class with downsampled majority class
    bank_df_downsampled = pd.concat([bank_df_majority_downsampled, bank_df_minority])

    # Display new class counts
    print("Percentage of each class after Undersampling:\n", bank_df_downsampled.y.value_counts(normalize=True), '\n')

    targets = bank_df_downsampled['y'].values
    features = pd.get_dummies(bank_df_downsampled.drop(['y'], axis=1)).values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    for train_index, test_index in sss.split(features, targets):
        X_train, X_test = features[train_index].T, features[test_index].T
        y_train, y_test = np.expand_dims(targets[train_index], axis=0), np.expand_dims(targets[test_index], axis=0)

    m_train = X_train.shape[1]
    m_test = X_test.shape[1]
    num_px = X_train[0].shape[0]


    print ("Number of training examples: m_train = " + str(m_train))
    print ("Number of testing examples: m_test = " + str(m_test))
    print ("Each training example is of size: " + str(num_px))
    print ("train_set_x shape: " + str(X_train.shape))
    print ("train_set_y shape: " + str(y_train.shape))
    print ("test_set_x shape: " + str(X_test.shape))
    print ("test_set_y shape: " + str(y_test.shape) + "\n")

    LR = LogisticRegression()
    # d = LR.model(X_train, y_train, X_test, y_test, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    # LR.plot_learning_curve(d)

    learning_rates = [0.1, 0.01]
    models = {}
    for i in learning_rates:
        print("Learning rate is: " + str(i))
        models[str(i)] = LR.model(X_train, y_train, X_test, y_test, num_iterations=10000, learning_rate=i,
                               print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
