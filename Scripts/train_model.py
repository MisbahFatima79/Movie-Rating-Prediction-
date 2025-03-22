import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset (replace 'data.csv' with your actual dataset)
df = pd.read_csv('data.csv')

# Define features and target variable
X = df[['feature1', 'feature2']]  # Replace with actual column names
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training complete!")

