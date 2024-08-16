import numpy as np

# Define the paths to the input files and the output files
file1_path = 'Figures/test_clean_image/acorn1.JPEG/gradient_positions_0_35.txt'
file2_path = 'Figures/test_clean_image/acorn1.JPEG/gradient_positions_1_35.txt'
output_txt_path = 'acorn_1_pixel_35.txt'
output_npy_path = 'acorn_1_pixel_35.npy'

# Read data from the first file
with open(file1_path, 'r') as f:
    data1 = [line.strip() for line in f.readlines()]

# Read data from the second file
with open(file2_path, 'r') as f:
    data2 = [line.strip() for line in f.readlines()]

# Merge the data from both files
merged_data = data1 + data2

# Remove duplicates by converting to a set and back to a list
unique_data = list(set(merged_data))

# Sort the unique data (optional)
unique_data.sort()

# Write the unique data to a new text file
with open(output_txt_path, 'w') as f:
    for item in unique_data:
        f.write(f"{item}\n")

# Convert the unique data to a NumPy array
unique_array = np.array([list(map(int, line.split(','))) for line in unique_data])

# Save the unique data to a NumPy .npy file
np.save(output_npy_path, unique_array)

loaded_array = np.load('acorn_1_pixel_35.npy')
print(loaded_array)
print(f"Unique data saved to {output_txt_path} and {output_npy_path}")