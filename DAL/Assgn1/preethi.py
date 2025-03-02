import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_frame = pd.read_csv('Christmas1.csv')  

# cleansing to remove any invalid data points 
cleaned_df = data_frame[(data_frame["X'"] >= 0) & (data_frame["X'"] <= 100) & (data_frame["Y'"] >= 0) & (data_frame["Y'"] <= 100)]

# discretize the data to a 1000x1000 boolean matrix
grid_size = 1000
boolean_grid = np.zeros((grid_size, grid_size), dtype=bool)


x_scaled = (cleaned_df["X'"] * (grid_size / 100)).astype(int)
y_scaled = (cleaned_df["Y'"] * (grid_size / 100)).astype(int)
boolean_grid[y_scaled, x_scaled] = True

# Rotate the matrix by 90 degrees clockwise (Transpose and then reverse columns)
rotated_grid = np.flip(boolean_grid.T, axis=1)

# Flip the original matrix horizontally (Reverse the columns)
flipped_grid = np.flip(boolean_grid, axis=1)

# Extract the coordinates f
rotated_coords_v1 = np.column_stack(np.nonzero(rotated_grid))
flipped_coords_v1 = np.column_stack(np.nonzero(flipped_grid))

# Plot the scatter plot for the original, rotated, and flipped images
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original image
original_coords_v1 = np.column_stack(np.nonzero(boolean_grid))
axes[0].scatter(original_coords_v1[:, 1], grid_size - original_coords_v1[:, 0], s=1)
axes[0].set_title('Original Image')
axes[0].invert_yaxis()

# Rotated image
axes[1].scatter(rotated_coords_v1[:, 1], grid_size - rotated_coords_v1[:, 0], s=1)
axes[1].set_title('Rotated Image')
axes[1].invert_yaxis()

# Flipped image
axes[2].scatter(flipped_coords_v1[:, 1], grid_size - flipped_coords_v1[:, 0], s=1)
axes[2].set_title('Flipped Image')
axes[2].invert_yaxis()

plt.show()
