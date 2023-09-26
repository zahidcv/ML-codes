import numpy as np
num_samples = 100
num_features = 3

X = 2 * np.random.rand(num_samples, num_features)
true_theta = np.array([3, 1.5, 2])
noise = np.random.randn(num_samples)
y = X.dot(true_theta) + noise

X = np.c_[np.ones((X.shape[0], 1)), X]

theta = np.ones((num_features+1))
print(theta.shape)
# theta = np.array(X.shape[1])
