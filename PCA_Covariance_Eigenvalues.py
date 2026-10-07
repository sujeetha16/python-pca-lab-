import pandas as pd
import numpy as np

# Load dataset
file_path = r"C:\Users\ADMIN\Downloads\dataset.csv"
df = pd.read_csv(file_path)

# Select numerical columns only
X = df.select_dtypes(include="number")

# Remove columns containing only missing values
X = X.dropna(axis=1, how="all")

# Fill missing values with column mean
X = X.fillna(X.mean())

# Calculate covariance matrix
cov_matrix = np.cov(X, rowvar=False)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Convert very small complex values to real values
eigenvalues = np.real_if_close(eigenvalues)

# Print covariance matrix
print("Covariance Matrix:")
print(cov_matrix)

# Print eigenvalues
print("\nEigenvalues:")
print(eigenvalues)
