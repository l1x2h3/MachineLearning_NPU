import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Read the Iris dataset from CSV
iris_df = pd.read_csv("iris.csv")

# Split the dataset into features and labels
X = iris_df.drop("Species", axis=1)
y = iris_df["Species"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
# Train the classifier
knn.fit(X_train, y_train)

# Save the model parameters to a txt file
with open('knn_model_parameters.txt', 'w') as file:
    file.write('KNN Classifier Model Parameters:\n')
    file.write(f'Number of neighbors: {knn.n_neighbors}\n')
    file.write(f'Weights: {knn.weights}\n')
    file.write(f'Metric: {knn.metric}\n')

# Save the trained model to a file
joblib.dump(knn, 'knn_trained_model.pkl')