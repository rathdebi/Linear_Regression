# linear regression model representation
import math, copy
import numpy as np
import matplotlib.pyplot as plt
# from datetime import datetime
# from plot_utils import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
plt.style.use("tableau-colorblind10")

def load_data(scan_dir):
    """
    load data from path given if exists
    params:
        scan_di (os.path) : input path to file name

    returns:
        x,y data as numpy array
    """
    with open(scan_dir) as file:
        data = np.loadtxt(file, delimiter=",", skiprows=1)
        x = data[:, :4]
        y = data[:, 4]
    return x, y


def create_data(x,y):
    """
    creates data required for the model
    params:
      x (list): training data, m examples
      y (list): target values, m examples
    returns:
      x, y as numpy array
    """
    x_train = np.array(x)
    y_train = np.array(y)
    return x_train, y_train


def normalize_features_using_zscore(x):
    """
    computes x, zcore normalized by column

    params:
        x (ndarray (m,n))     : input data, m examples, n features

    returns:
        x_norm (ndarray (m,n)): input normalized by feature
        mu (ndarray (n,))     : mean of each feature
        sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean(x.copy(), axis=0)  # mu will have shape (n,)

    # find the standard deviation of each column/feature
    sigma = np.std(x.copy(), axis=0)  # sigma will have shape (n,)

    # element-wise, subtract mu for that feature from each example,
    # divide by std deviation for that feature
    x_mean = (x - mu)
    x_norm = (x - mu) / sigma

    return (x, x_mean, x_norm, mu, sigma)

def plot_feature_by_target(x, y, features, y_label):
    """
    plot each feature by target variable to derive relation
    params:
        x (ndarray (m,)): data m examples
        y (ndarray (m,)): target values
        features (list) : feature names (used for plotting)
        y_label (string): target label in y-axis (used for plotting)
    returns:
        plot by each feature
    """
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for ax_num in range(len(ax)):
        ax[ax_num].scatter(x[:, ax_num], y)
        ax[ax_num].set_xlabel(features[ax_num])
        ax[ax_num].set_ylabel(y_label)
    plt.show()

def plot_distribution_using_normalization(x,
                                          features,
                                          title,
                                          y_label):
    """
    plot each feature to see distribution before, between and after
    normalization.
    params:
        x list((ndarray (m,))): list of data m examples
        features (list) : feature names (used for plotting)
        title (list)    : title names used for each plot
        y_label (string): target label in y-axis (used for plotting)

    returns:
        plot to see feature distribution
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    for ax_num in range(len(ax)):
        if ax_num == 0:
            x, title = x[0], title[0]
        if ax_num == 1:
            x, title = x[1], title[1]
        if ax_num == 2:
            x, title = x[2], title[2]
        ax[ax_num].scatter(x[:, ax_num], x[:,3])
        ax[ax_num].set_xlabel(features[0])
        ax[ax_num].set_ylabel(features[3])
        ax[ax_num].set_title(title)
        ax[ax_num].axis(y_label)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("distribution of features before, during, after normalization")
    plt.show()




# ax[0].scatter(x[:,0], x[:,3])
# ax[0].set_xlabel(feature_names[0]); ax[0].set_ylabel(feature_names[3]);
# ax[0].set_title("un-normalized")
# ax[0].axis('equal')
#
# ax[1].scatter(X_mean[:,0], X_mean[:,3])
# ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
# ax[1].set_title(r"X - $\mu$")
# ax[1].axis('equal')
#
# ax[2].scatter(X_norm[:,0], X_norm[:,3])
# ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
# ax[2].set_title(r"Z-score normalized")
# ax[2].axis('equal')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.suptitle("distribution of features before, during, after normalization")
# plt.show()
#

def plot_data(x_train,
              y_train,
              predictions,
              title,
              x_label,
              y_label,
              predicted,
              marker= "x",
              color= ["r","b"],
              label= ["actual", "prediction"]):
    """
    plot x_train and y_train in x-y scatter plot , prediction in line plot
    params:

      x_train (ndarray (m,)): data m examples
      y_train (ndarray (m,)): target values
      prediction (ndarray (m,)): predicted target values
      predicted boolean: prediction flag set to False as default, set to True while plotting
      marker string: selected marker pointer used in plotting ("x")
      color list: selected color pointer used in plotting ("r"- red for training, "b"- blue for prediction)
      label list: selected label names for plotting ( for x-y scatter plot- actual, for line plot- prediction)

    returns:
      data and target values plot accordingly
    """
    plt.scatter(x_train, y_train, marker=marker, c=color[0], label=label[0])
    if predicted:
        plt.plot(x_train, predictions, c=color[1], label=label[1])
    else:
        pass
    # set title
    plt.title(title)
    # set x_label
    plt.xlabel(x_label)
    # set y_label
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
#
#
# def compute_model_output(x, w, b):
#     """
#     computes the prediction of a linear model
#     params:
#
#       x (ndarray (m,)): data, m examples
#       w,b (scalar)    : model parameters
#
#     returns:
#       y (ndarray (m,)): target values
#     """
#     x = x.copy()
#     training_example = x.shape[0]
#     predictions = np.zeros(training_example)
#     for i in range(training_example):
#         predictions[i] = w * x[i] + b
#
#     return predictions
#
# # set w = 2.2 and b = 2 and test your prediction, it looks good
#
# # predictions = compute_model_output(x_train, 2.2, 2)
#
# # plot_data(x_train, y_train, predictions,"expected salary", "years of experience", "ctc in lpa",True)
#
# # print(f"our model f(w,b) predictions: {predictions}")
#
#
# # this above model is a simple linear model representation having random
# # weights and bias initialized to detail out a meaningful interpretation
# # actually for each training example (experience to salary) mapping the
# # function f(w,b) would incur some cost , termed as error. the error
# # associated with all m training example can be summed up to define the cost function.
#
#
# def compute_cost(x_train, y_train, w, b):
#     """
#     computes cost of linear regression function f(w,b) that fits data points x, y
#
#
#     params:
#       x_train (ndarray (m,)): data, m examples
#       y_train  (ndarray (m,)): target values
#       w,b (scalar)    : model parameters, w- weight, b- bias
#
#     returns:
#         total_cost (float): the cost of using w,b as the parameters for linear regression
#                to fit the data points in x and y
#     """
#     # number of training examples
#     training_examples = x_train.copy().shape[0]
#     cost_sum = 0
#     for example in range(training_examples):
#         predictions = w * x_train[example] + b
#         cost = (predictions - y_train.copy()[example]) ** 2
#         cost_sum += cost
#     total_cost = (1 / (2 * training_examples)) * cost_sum
#
#     return total_cost
#
# # w,b = 2.2,2 # after few hit and trails
# # total_cost = compute_cost(x_train, y_train, 2.2, 2)
# # print(f"total cost without gradient descent: {w}, bias: {b} in fitting training data: {total_cost:.2f}")
#
#
#
# # our goal is to find a model f(w,b)(x) = w*x + b, with parameters w,b which will
# # accurately or nearly predict expected salary values given an input experience in years.
# # the cost is a measure of how accurate the model is fitting the training data.
# # the algorithm that helps us find w, b values that helps us to minimize cost
# # gradient descent is often described as an algorithm,
# # 1- w = w - alpha * d/dw[(J(w,b)], derivative of cost w.r.t w
# # 2- b = b - alpha * d/db[(J(w,b)], derivative of cost w.r.t b
# # where parameters w, b are updated simultaneously, alpha is the learning date
# # d/dw[(J(w,b)] = 1/m * sum[(f(w,b)* x - y]*x # using partial derivative to update
# # d/db[(J(w,b)] = 1/m * sum[(f(w,b) * x -y] # using partial derivative to update
#
# def compute_gradient(x, y, w, b):
#     """
#     computes the gradient for linear regression
#     params:
#       x (ndarray (m,)): data, m examples
#       y (ndarray (m,)): target values
#       w,b (scalar)    : model parameters
#     returns:
#       dj_dw (scalar): gradient of the cost w.r.t. the parameters w
#       dj_db (scalar): gradient of the cost w.r.t. the parameter b
#      """
#     training_example = x.copy().shape[0]
#     dj_dw = dj_db = 0
#     for example in range(training_example):
#         predictions = w * x[example] + b
#         dj_dw_example= (predictions - y.copy()[example]) * x[example]
#         dj_db_example= predictions - y.copy()[example]
#         dj_db += dj_db_example
#         dj_dw += dj_dw_example
#     dj_dw = dj_dw / example
#     dj_db = dj_db / example
#
#     return dj_dw, dj_db
#
# # dj_dw, dj_db = compute_gradient(x_train, y_train,w, b) # get gradients
# # print(f"gradients computed for weight : {dj_dw:.2f} and bias : {dj_db:.2f}")
# # plt_gradients(x_train,y_train, compute_cost, compute_gradient)
# # plt.show()

def compute_gradient_descent(x,
                             y,
                             w_in,
                             b_in,
                             cost_function,
                             gradient_function,
                             alpha,
                             num_iters,):
    """
    performs gradient descent to fit w,b. updates w,b by taking
    num_iters gradient steps with learning rate alpha

    params:
      x (ndarray (m,))  : Data, m examples
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters
      alpha (float):     learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to compute cost
      gradient_function: function to call to compute gradient

    returns:
      w (scalar): updated value of parameter after running gradient descent
      b (scalar): updated value of parameter after running gradient descent
      history (List): history of cost values
      """
    # avoid modifying global w_in by reference
    # array to store cost J and w's at each iteration primarily for graphing later
    w = copy.deepcopy(w_in)
    b = b_in
    save_interval = np.ceil(num_iters / 10000)
    history = {"cost": list(),
            "params": list(),
            "gradients": list(),
            "iterations": list(),}

    print(f"iteration cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  ")
    print(f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|")
    for iteration in range(num_iters):
        # compute gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b,True)

        # Update Parameters using equation (3) above
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # save cost J at each iteration
        if iteration == 0 or iteration % save_interval == 0:  # prevent resource exhaustion
            history["cost"].append(cost_function(x, y, w, b, True))
            history["params"].append([w, b])
            history["gradients"].append([dj_dw, dj_db])
            history["iterations"].append(iteration)

        # print cost every at intervals 10 times or as many iterations if < 10
        if iteration % math.ceil(num_iters / 10) == 0:
            # print(f"iteration: {iteration:4d}, cost: {J_history[-1]:8.2f}")
            cost = cost_function(x, y, w, b,True)
            print(f"{iteration:9d} {cost:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}")

    return w, b, history # w, b and history for graphing


def run_gradient_descent(x, y, iterations=1000, alpha=1e-6):
    """run gradient descent for a specific iteration and alpha value
    args:
        x (ndarray (m,))  : data, m examples
        y (ndarray (m,))  : target values
        iterations (int)  : number of iterations
        alpha (float)     : learning rate , positive integer
    returns:
        w (scalar): updated value of parameter after running gradient descent
        b (scalar): updated value of parameter after running gradient descent
        w (scalar): updated value of parameter after running gradient descent
        hist_out (list): history of cost values
    """
    num_rows, num_features = x.shape
    # initialize parameters
    initial_w = np.zeros(num_features)
    initial_b = 0

    # run gradient descent
    w_out, b_out, hist_out = compute_gradient_descent(x,
                                                     y,
                                                     initial_w,
                                                     initial_b,
                                                     compute_cost_loop_or_vectorized,
                                                     compute_gradient_loop_or_vectorized,
                                                     alpha,
                                                     iterations,)
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.2f}")

    return (w_out, b_out, hist_out)



#
# # initialize parameters
# w_init = b_init = 0
# # some gradient descent settings
# iterations, tmp_alpha = 10000, 1.0e-2
# # run gradient descent
# w_final, b_final, J_hist, p_hist = compute_gradient_descent(x_train ,
#                                                             y_train,
#                                                             w_init,
#                                                             b_init,
#                                                             tmp_alpha,
#                                                             iterations,
#                                                             compute_cost,
#                                                             compute_gradient)
# print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
# print(f"our linear model representation salary = {w_final:.2f} * experience + {b_final:.2f}")
# print(".......prediction with unseen data using linear regression model......")
# print(f"15 years experience predicted salary {w_final*11 + b_final:0.1f} lpa.")
# print(f"9.5 years experience predicted salary {w_final*6.5 + b_final:0.1f} lpa.")
# print(f"7.6 years experience predicted salary {w_final*8.8 + b_final:0.1f} lpa.")


# we have seen that our linear representation model is able to predict
# using one feature . going further we will see how can we use multiple features
# to predict target value. this is obvious that a model will have many features
# instead only one. in this case we will use and tweak that same understanding
# of single prediction to make multiple feature prediction.

# our cost, predict, gradient and gradient descent function will have some
# changes , very minor, shuttle changes. the changes can be implemented
# via vectorization. in simple terms vectorization helps us by extending
# the use of computer hardware to run parallel task at the same time to avoid
# explicit for loops which took significant amount of time.


# our model representation of multiple feature , f(w, b)(X) = w0 * x0 + w1 * x1 +....+ wn * xn + b
# if you see carefully observe we have just extended that line of equation. nothing else.
# we can write it like this-- f(w,b)(X) = W . X + b [ using .dot product]
# in other words we can say that single prediction element by element with many features

# create data with many features-- experience, rating, offers, domain
# rating - 1 to 5 (1 low and 5 high)
# domain - 1 as SE/ST, 2 as DB, 3 as DE, 4 as AI/ML
x_train = np.array([[1, 2, 0, 1],
                    [2, 3, 0, 1,],
                    [3, 4, 0, 2,],
                    [4, 4, 1, 3,],
                    [3, 3, 2, 2,],
                    [6, 4, 1, 3,],
                    [7, 4, 2, 3,],
                    [8, 4, 2, 4,],
                    [9, 4, 2, 4,],
                    [10, 4, 3, 4]])

y_train = np.array([3.0, 5.0, 8.2, 12.1, 9.2, 15.0, 17.3, 20.1, 21.7, 24.3,])
x_train, y_train = create_data(x_train,y_train)
print(f"\nnumber of training examples : {x_train.shape[0]} \nnumber of features : {x_train.shape[1]}")

def predict_single_loop_or_vector(x, w, b,run_vectorized= False):
    """
    single row predict using linear regression

    args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):  model parameter
      run_vectorized (boolean): run type flag, default set to false

    returns:
      predictions (scalar):  prediction

    pseudo version--loop
    # iterate through all examples/records
    # get prediction for each row
    # add prediction, one by one
    # add bias, the base term
    # return prediction

    pseudo version--vector
    # element wise prediction
    # all prediction at hardware level/parallel
    # single instruction multiple data (SIMD)
    # add bias term at the end
    # return prediction
    """
    # avoid getting x being changed due to reference
    x = x.copy()
    if not run_vectorized:
        print("----prediction using looping mechanism----")
        num_rows = x.shape[0]
        predictions = 0
        for row in range(num_rows):
            prediction_row = x[row] * w[row]
            predictions = predictions + prediction_row
        predictions = predictions + b

    else:
        print("----prediction using vectorized implementation----")
        predictions = np.dot(x, w) + b

    return predictions

# for single example prediction, get data from x_train
# get a row from our training data, that is a row vector with all features
x_vec = x_train[0,:] # first row/record
print(f"x_vec shape: {x_vec.shape}, x_vec type : {type(x_vec)}")

# make a prediction , using w_init and b_init as close
w_init = np.array([1.01, 1.02, 1.04, 1.03,]) # 4 elements
b_init = 0.20 # scaler value
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")
predictions = predict_single_loop_or_vector(x_vec, w_init, b_init, True)
print(f"single example: {x_vec}\nactual salary: {y_train[0]:.2f}\npredicted salary: {predictions:.2f}")


def compute_cost_loop_or_vectorized(x, y, w, b, run_vectorized=False):
    """
    compute cost with one feature or n features [loop or vectorized implementation]
    args:
      x (ndarray (m,n)): data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    returns:
      cost (scalar): cost
    """
    num_rows, cost = x_train.copy().shape[0], 0
    if not run_vectorized:
        for row in range(num_rows):
            predictions = w * x[row] + b
            cost += (predictions - y.copy()[row]) ** 2
        cost = (1 / (2 * num_rows)) * cost

    else:
        for row in range(num_rows):
            predictions = np.dot(x[row], w) + b  # scalar (see np.dot)
            cost += (predictions - y[row]) ** 2  # scalar
        cost = cost / (2 * num_rows)  # scalar

    return cost

# compute and display cost using our pre-chosen optimal parameters.
cost = compute_cost_loop_or_vectorized(x_train, y_train, w_init, b_init,run_vectorized=True)
print(f"cost at optimal {w_init} : {cost:.2f}")


# as we are using multiple features , so to optimize that cost we will
# use gradient descent to update our model parameters w and b simultaneously.
# our intuition behind cost parameter b would remain the same looping through
# all training examples , as a scaler value. similarly, parameter w would loop
# through all features computed and accumulated at each w.

def compute_gradient_loop_or_vectorized(x, y, w, b, run_vectorized=False):
    """
    computes the gradient for linear regression using vectorised implementation
    args:
      x (ndarray (m,n)): data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
      run_vectorized (bool): run type flag, default set to false

    returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    num_rows, num_features = x.copy().shape
    if not run_vectorized:
        dj_dw = dj_db = 0
        for row in range(num_rows):
            predictions = w * x[row] + b # loop through number of rows
            dj_dw_row = (predictions - y.copy()[row]) * x[row]
            dj_db_row = predictions - y.copy()[row]
            dj_db += dj_db_row
            dj_dw += dj_dw_row

    else:
        dj_dw, dj_db = np.zeros((num_features,)), 0
        for row in range(num_rows):  # loop through number of rows
            error = (np.dot(x[row], w) + b) - y[row]
            for feature in range(num_features):  # loop through number of features
                dj_dw[feature] += error * x[row, feature]
            dj_db += error
    dj_dw /= num_rows
    dj_db /= num_rows

    return dj_dw, dj_db


# compute and display gradient
tmp_dj_dw, tmp_dj_db = compute_gradient_loop_or_vectorized(x_train,
                                                           y_train,
                                                           w_init,
                                                           b_init,
                                                           run_vectorized=True)
print(f"dj_db at initial w,b: {tmp_dj_db}\ndj_dw at initial w,b: {tmp_dj_dw}")


# now that gradient is computed, so the same step needs to be followed with
# an optimal way to get the global optima, termed as gradient descent. in this
# case cost function and gradient function will be used to compute final optimal
# parameters where our cost is minimum. this is known as batch gradient, that
# effectively learns cost (theta), updates it simultaneously using learning rate.
# we will use the earlier version of gradient descent to compute gradient descent.


# initialize parameters w_init and b_init to 0
initial_w, initial_b = np.zeros_like(w_init), 0
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7

# run gradient descent
w_final, b_final, J_history = compute_gradient_descent(x_train,
                                                     y_train,
                                                     initial_w,
                                                     initial_b,
                                                     compute_cost_loop_or_vectorized,
                                                     compute_gradient_loop_or_vectorized,
                                                     alpha,
                                                     iterations,)
print(f"final b,w found by gradient descent: {b_final:.3f}, {w_final}")

num_rows = x_train.shape[0]
for row in range(num_rows):
    print(f"predicted value: {np.dot(x_train[row], w_final) + b_final}, actual value: {y_train[row]}")












