import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_flat = X_train.reshape(-1, 28*28).astype(np.float32) / 255.0
X_test_flat = X_test.reshape(-1, 28*28).astype(np.float32) / 255.0

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Initialize and train the SVM model with RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma='scale')  # 'scale' is a good default for gamma
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_svm = model.predict(X_test_scaled)

# Evaluate the model
accuracy_svm = accuracy_score(y_test, y_pred_svm)


# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_continuous = model.predict(X_test_scaled)

# Convert continuous predictions to class labels
y_pred_linear = np.round(y_pred_continuous).astype(int)  # Round to nearest integer

# Ensure predictions are within the valid class range (0-9)
y_pred_linear = np.clip(y_pred_linear, 0, 9)

# Evaluate the model using accuracy
accuracy_linear = accuracy_score(y_test, y_pred_linear)

# n_clusters = 10  # Number of clusters for MNIST (digits 0-9)
# kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
# kmeans.fit(X_train_scaled)

# # Predict clusters for the test set
# y_pred_kmeans = kmeans.predict(X_test_scaled)

# # Evaluate clustering by mapping clusters to true labels
# def map_clusters_to_labels(true_labels, predicted_labels):
#     # Create confusion matrix
#     cm = confusion_matrix(true_labels, predicted_labels)
#     # Solve the linear sum assignment problem (Hungarian algorithm) to find the best label mapping
#     row_ind, col_ind = linear_sum_assignment(-cm)
#     # Create mapping from cluster index to true label
#     label_mapping = dict(zip(row_ind, col_ind))
#     return label_mapping

# # Find the best label mapping
# label_mapping = map_clusters_to_labels(y_test, y_pred)

# # Map predicted labels to true labels
# y_pred_mapped = np.array([label_mapping[label] for label in y_pred])

# # Calculate accuracy
# accuracy_kmeans = accuracy_score(y_test, y_pred_mapped)

def main(): 
    model_best =''
    if accuracy_svm > accuracy_linear:
        model_best = 'svm'
    else :
        model_best = 'linear'
    print(model_best)

# __name__ 
if __name__=="__main__": 
    main() 