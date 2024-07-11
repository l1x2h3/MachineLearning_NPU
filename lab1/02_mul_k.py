import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Read the Iris dataset from CSV
iris_df = pd.read_csv("iris.csv")

# Split the dataset into features and labels
X = iris_df.drop("Species", axis=1)
y = iris_df["Species"]

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize K values
k_values = [1, 3, 5, 7]
accuracies = []

for k in k_values:
    # Initialize the KNN classifier with k value
    knn = KNeighborsClassifier(n_neighbors=k)
    # Train the classifier
    knn.fit(X_train, y_train)
    # Predict the labels for the test data
    y_pred = knn.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(f"Accuracy for k={k}: {accuracy}")

# Create a DataFrame to display K values and their accuracies
results_df = pd.DataFrame({'K': k_values, 'Accuracy': accuracies})
print(results_df)