import numpy as np

# Sample dataset (features and labels)
X = np.array([[2.5, 3.0], [2.0, 2.0], [3.5, 4.0], [3.0, 3.5], [4.0, 2.5]])
y = np.array([0, 0, 1, 1, 0])  # Labels (0: Class A, 1: Class B)

# Define a Decision Tree Node
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the feature
        self.left = left  # Left subtree (child node)
        self.right = right  # Right subtree (child node)
        self.value = value  # Class label (if leaf node)

# Define a Decision Tree Classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Stop recursion conditions
        if depth == self.max_depth or num_classes == 1 or num_samples < 2:
            return DecisionTreeNode(value=np.argmax(np.bincount(y)))

        # Find the best split
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return DecisionTreeNode(value=np.argmax(np.bincount(y)))

        feature_index, threshold = best_split
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return DecisionTreeNode(feature_index, threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None

        num_parent = [np.sum(y == c) for c in np.unique(y)]
        best_gini = 1.0 - sum((n / num_samples) ** 2 for n in num_parent)
        best_split = None

        for feature_index in range(num_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature_index], y)))
            num_left = [0] * num_classes
            num_right = num_parent.copy()

            for i in range(1, num_samples):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(num_classes)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (num_samples - i)) ** 2 for x in range(num_classes)
                )
                gini = (
                    (i / num_samples) * gini_left
                    + ((num_samples - i) / num_samples) * gini_right
                )
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, (thresholds[i] + thresholds[i - 1]) / 2)
        return best_split

    def predict(self, X):
        return [self._predict_tree(x, self.root) for x in X]

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_tree(x, node.left)
        return self._predict_tree(x, node.right)

# Create and train the decision tree classifier
tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X, y)

# Sample data for prediction
X_test = np.array([[2.8, 3.2], [3.7, 4.1]])

# Make predictions
predictions = tree.predict(X_test)

# Print predictions
for i, pred in enumerate(predictions):
    print(f"Sample {i + 1}: {'Class A' if pred == 0 else 'Class B'}")
