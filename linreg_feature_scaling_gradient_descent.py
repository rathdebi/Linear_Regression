import numpy as np

import matplotlib.pyplot as plt
plt.style.use("./deeplearning.mplstyle")
from linear_regression import (run_gradient_descent,
                                load_data,
                                plot_feature_by_target,
                                normalize_features_using_zscore,
                                plot_distribution_using_normalization,)
from utils.lab_utils_multi import norm_plot, plt_equal_scale, plot_cost_i_w


# load the dataset
x_train, y_train = load_data("C:\\Users\\2000080516\\Downloads\\machine_learning\\linear_regression\\data\\houses.txt")
feature_names = ["size_in_sqft", "num_bedrooms", "num_floors", "age_of_house",]
print(x_train, y_train)
print("data loaded successfully!")

# plot each feature with target to see relation
# plot_feature_by_target(x_train, y_train, feature_names, "price in 1000's")

# we know that finding a learning rate can be tough, as it controls size of the update
# on the parameters (w, b) -- let us see a few settings on alpha.
# alpha settings -- start with 9.9e-7, 9e-7, 1e-7 and so on. plug and play
# you will notice cost is gradually decreasing, leading to a global optima.
# but rather very slow update.

_, _, history = run_gradient_descent(x_train, y_train, 10, alpha = 1e-7)

# the reason behind our slow convergence is that our features are not scaled
# in one scale. as we can our features have different range of values. that way
# one step cost/gradient update would make other one go wrong in a big margin.
# in this case e need to ensure that all our features are in the same range.
# normalization-- max-normalization, mean normalization, and z-score normalization.
# three different techniques:-- to explain further refer below explanation
# dividing each positive feature by its maximum value using x/max
# or more generally, rescale each feature by both its minimum and maximum values using (x-min)/(max-min).
# both ways normalize features to the range of -1 and 1, where the former method works for positive features
# which is simple and serves well and the latter method works for any features.
# Mean normalization: x - mean/(max - min)
# Z-score normalization: x - mean/ sigma ,where mean- average, sigma- std. deviation
# Z-score normalization will make all features mean 0 and std. deviation as 1.


"""distribution of features before, during, after normalization"""
x_train, x_mean, x_norm, mu, sigma = normalize_features_using_zscore(x_train)
fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(x_train[:,0], x_train[:,3])
ax[0].set_xlabel(feature_names[0]); ax[0].set_ylabel(feature_names[3]);
ax[0].set_title("un-normalized")
ax[0].axis('equal')

ax[1].scatter(x_mean[:,0], x_mean[:,3])
ax[1].set_xlabel(feature_names[0]); ax[0].set_ylabel(feature_names[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(x_norm[:,0], x_norm[:,3])
ax[2].set_xlabel(feature_names[0]); ax[0].set_ylabel(feature_names[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()


fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],x_train[:,i],)
    ax[i].set_xlabel(feature_names[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()

fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],x_norm[:,i],)
    ax[i].set_xlabel(feature_names[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features after normalization")
plt.show()



w_norm, b_norm, hist = run_gradient_descent(x_norm, y_train, 1000, 1.0e-1,)

#predict target using normalized features
num_rows = x_norm.shape[0]
y_pred = np.zeros(num_rows)
for num in range(num_rows):
    y_pred[num] = np.dot(x_norm[num], w_norm) + b_norm


# plot predictions and targets versus original features
fig,ax = plt.subplots(1,4,figsize=(12, 3),sharey=True)
for ax_num in range(len(ax)):
    ax[ax_num].scatter(x_train[:,ax_num],y_train, label = "target")
    ax[ax_num].set_xlabel(feature_names[ax_num])
    ax[ax_num].scatter(x_train[:,ax_num],y_pred,color="orange", label = 'predict')
ax[0].set_ylabel("Price");
ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()


# First, normalize given example.
new_house_data = np.array([1200, 3, 1, 40])
new_house_norm = (new_house_data - mu) / sigma
print(new_house_norm)
new_house_price_predict = np.dot(new_house_norm, w_norm) + b_norm
print(f"predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${new_house_price_predict*1000:0.0f}")
plt_equal_scale(x_train, x_norm, y_train)
























