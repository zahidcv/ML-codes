import numpy as np
import matplotlib.pyplot as plt
def gradient_descent(X, y, theta, alpha, num_iterations):

  J_all = []

  for i in range(num_iterations):
    # Compute the predictions.
    # print(X.shape, y.shape)
    h = np.dot(X, theta)

    # Compute the cost function.
    J = np.sum((h - y)**2) / (2 * X.shape[0])

    # Compute the gradients.
    gradients = np.dot(X.T, (h - y)) / X.shape[0]

    # Update the model parameters.
    theta -= alpha * gradients

    # Store the cost function value for this iteration.
    J_all.append(J)

  return theta, J_all


num_samples = 5
num_features = 3

X = 2 * np.random.rand(num_samples, num_features)
true_theta = np.array([3, 1.5, 2])
noise = 10
y = X.dot(true_theta) + noise

X = np.c_[np.ones((X.shape[0], 1)), X]

theta = np.ones((num_features+1))
alpha = 0.1

num_iterations = 10000

theta, J_all = gradient_descent(X, y, theta, alpha, num_iterations)
print(noise, true_theta)

print(theta)
plt.plot(num_iterations, J_all)