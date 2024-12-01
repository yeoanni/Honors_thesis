

import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt

np.random.seed(0)
# Define P (10x10 matrix)
P = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        P[i, j] = i + j + 2  # Add +2 for Python index alignment

# Define G (10x2 matrix)
G = np.column_stack((np.arange(1, 11), np.arange(1, 11)**2))  # 1 to 10 and their squares

# Set reduced rank
r = 5

# Perform SVD and reduce dimensions
  # Set desired rank for dimensionality reduction
U, S, Vt = svd(P, full_matrices=False)  # Economy SVD for potentially non-square P
S_matrix = np.diag(S)
V = Vt.T

# Cumulative energy criterion to select r dynamically
# cumulative_energy = np.cumsum(S) / np.sum(S)
# r = np.argmax(cumulative_energy >= 0.95) + 1  # Find first r for 95% energy

# Select the first r rows and columns
U_r = U[:r, :r]          # First r rows and columns of U
S_r = np.diag(S[:r])     # First r singular values (as a diagonal matrix)
V_r = V[:r, :r]         # First r rows and columns of V
V_rt = V_r.T

Psmall = U_r @ S_r @ V_rt  # Rank-r approximation of P

# Truncate negative values and normalize rows
# Psmall = np.maximum(Psmall, 0)  # Truncate negative values
Psmall = Psmall / Psmall.sum(axis=1, keepdims=True)  # Normalize rows

# Calculate weights matrix directly from V
Weights = V[:r, :]  

# Calculate Gsmall as a weighted average in reduced space
Gsmall = Weights @ G


# Calculate the range of the blue points (original grid G)
blue_min = np.min(G, axis=0)  # Minimum values in each dimension [min_x, min_y]
blue_max = np.max(G, axis=0)  # Maximum values in each dimension [max_x, max_y]

# Check which red points (reduced grid Gsmall) fall within the blue range
within_x_range = (Gsmall[:, 0] >= blue_min[0]) & (Gsmall[:, 0] <= blue_max[0])
within_y_range = (Gsmall[:, 1] >= blue_min[1]) & (Gsmall[:, 1] <= blue_max[1])

# Combine conditions to find points within both x and y range
within_range = within_x_range & within_y_range

# Count the number of red points within the range
num_within_range = np.sum(within_range)

# Calculate the ratio
ratio = num_within_range / Gsmall.shape[0]

# Print matrix
# Original matrix
print("P:")
print(P)

# SVD components
print("U:")
print(U)
print("S:")
print(S_matrix)
print("Vt:")
print(Vt)
print("V:")
print(V)

# Truncated components
print("U_r:")
print(U_r)
print("S_r:")
print(S_r)
print("V_r:")
print(V_r)

# Low-rank approximation
print("Psmall:")
print(Psmall)

# Weights matrix
print("Weights:")
print(Weights)

# Transformed grid
print("Gsmall:")
print(Gsmall)


# Display results
print(f"Initial P dimension: {P.shape}")
print(f"Initial G dimension: {G.shape}")

print(f"U dimension: {U.shape}")
print(f"S dimension: {S.shape}")
print(f"S_matrix dimension: {S_matrix.shape}")
print(f"V dimension: {V.shape}")

print(f"U_r dimension: {U_r.shape}")
print(f"S_r dimension: {S_r.shape}")
print(f"V_r dimension: {V_r.shape}")

print(f"Psmall dimension (low-rank approximation): {Psmall.shape}")
print(f"Psmall dimension (normalized): {Psmall.shape}")
print(f"Weights dimension: {Weights.shape}")
print(f"Gsmall dimension: {Gsmall.shape}")


print(f'Number of red points within the range of blue points: {num_within_range}')
print(f'Total number of red points: {Gsmall.shape[0]}')
print(f'Ratio (Preservation Metric): {ratio:.4f}')
print(f'Percentage within range: {ratio * 100:.2f}%')


# Visualization
plt.figure()
plt.scatter(G[:, 0], G[:, 1], color='blue', label='Original Grid (Blue)', s=10)  # Blue dots with size 10
plt.scatter(Gsmall[:, 0], Gsmall[:, 1], color='red', label='Reduced Grid (Red)', s=10)  # Red dots with size 10
plt.title('Original Grid (Blue) vs Reduced Grid (Red)')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.legend()
plt.show()
