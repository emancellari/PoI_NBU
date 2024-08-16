import numpy as np
import matplotlib.pyplot as plt
import os

def create_directory(directory):
    """
    Create directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_pixel_positions(file_path):
    """
    Read pixel positions from a text file. Each line should be in the format 'x,y'.
    """
    with open(file_path, 'r') as file:
        positions = [tuple(map(int, line.strip().split(','))) for line in file]
    return positions

def save_heatmap(positions, output_path, size=(224, 224), color='red', title='Heatmap', bgcolor='black'):
    """
    Generate and save a heatmap from pixel positions.
    """
    heatmap = np.zeros(size)
    for x, y in positions:
        heatmap[x, y] = 1  # Note: In images, the y-axis typically comes first.

    plt.imshow(heatmap, cmap='Greys', interpolation='nearest',origin='upper')
    plt.gca().set_facecolor(bgcolor)  # Set the background color

    plt.axis('off')  # Hide the axes
    for x, y in positions:
        plt.scatter(y, x, color=color, s=1)  # Overlay the points with the specified color
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def merge_positions(positions1, positions2, remove_duplicates=False):
    """
    Merge two lists of positions.
    If remove_duplicates is True, duplicates are removed.
    """
    merged = positions1 + positions2
    if remove_duplicates:
        merged = list(set(merged))
    return merged

def save_merged_heatmap(positions1, positions2, output_path, size=(224, 224), title='Merged Heatmap', bgcolor='black'):
    """
    Generate and save a heatmap showing positions from two lists with different colors.
    Red for positions1, blue for positions2.
    """
    heatmap = np.zeros(size)
    for x, y in positions1:
        heatmap[x, y] = 1
    for x, y in positions2:
        heatmap[x, y] = 2

    plt.imshow(heatmap, cmap='Greys', interpolation='nearest', origin='upper')
    plt.gca().set_facecolor(bgcolor)  # Set the background color
    plt.axis('off')  # Hide the axes
    for x, y in positions1:
        plt.scatter(y, x, color='red', s=0.5)
    for x, y in positions2:
        plt.scatter(y, x, color='blue', s=0.5)
    plt.title(title)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_pixel_positions(positions, file_path):
    """
    Save pixel positions to a text file.
    """
    with open(file_path, 'w') as file:
        for x, y in positions:
            file.write(f"{x},{y}\n")

# Directory to save results
directory = "Zoi_testing"
create_directory(directory)
percentage=35
image='acorn1'
# Read pixel positions
positions_ancestor = read_pixel_positions(f'Figures/test_clean_image/{image}.JPEG/gradient_positions_1_{percentage}.txt')
positions_target = read_pixel_positions(f'Figures/test_clean_image/{image}.JPEG/gradient_positions_0_{percentage}.txt')

# Save heatmap for ancestor category
#save_heatmap(positions_ancestor, os.path.join(directory, f'ancestor_heatmap_{image}_{percentage}.png'), color='red', title='', bgcolor='black')

# Save heatmap for target category
#save_heatmap(positions_target, os.path.join(directory, f'target_heatmap_{image}_{percentage}.png'), color='blue', title='', bgcolor='black')

# Merge positions keeping duplicates
#merged_positions_keep_duplicates = merge_positions(positions_ancestor, positions_target, remove_duplicates=False)
#save_pixel_positions(merged_positions_keep_duplicates, os.path.join(directory, f'merged_positions_keep_duplicates_{image}_{percentage}.txt'))
#save_merged_heatmap(positions_ancestor, positions_target, os.path.join(directory, f'merged_heatmap_keep_duplicates_{image}_{percentage}.png'), title='', bgcolor='black')

# Merge positions removing duplicates
merged_positions_remove_duplicates = merge_positions(positions_ancestor, positions_target, remove_duplicates=True)
save_pixel_positions(merged_positions_remove_duplicates, os.path.join(directory, f'merged_positions_remove_duplicates_{image}_{percentage}.txt'))
save_merged_heatmap(positions_ancestor, positions_target, os.path.join(directory, f'merged_heatmap_remove_duplicates_{image}_{percentage}.png'), title='', bgcolor='black')
