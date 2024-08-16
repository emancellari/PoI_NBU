import pytorchnet
from bagnet import zoi
import os

# Define valid image extensions
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')


def process_images_in_folder(folder_path, model, label, top_percentage, distance_threshold, category, resolution):
    """Process all valid image files in a folder using the zoi function."""
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        
        # Check if the file is an image based on its extension
        if image_file.lower().endswith(VALID_EXTENSIONS):
            try:
                # Process the image using the zoi function
                zoi(model, image_path, label, top_percentage, distance_threshold, category, resolution)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        else:
            print(f"Skipping non-image file: {image_file}")


def main():
    """Main function to initialize the model and process images in the specified folder."""
    
    # Initialize the model
    model = pytorchnet.bagnet33(pretrained=True)
    
    # Define folder path containing images
    folder_path = "test_clean_image"
    
    # Define category arrays for different labels
    labels_acorn = [988]
    labels_beetle = [306]
    
    # Get user input for various parameters
    top_percentage = int(input("Enter the top percentage value (e.g., 10): "))
    distance_threshold = int(input("Enter the distance threshold value for testing DBSCAN (e.g., 5): "))
    resolution = int(input("Enter resolution (0 for low resolution, 1 for high resolution): "))
    
    # Get user choice for category
    category_choice = int(input("Enter the category (0 for acorn, 1 for rhinoceros beetle): "))
    
    if category_choice not in [0, 1]:
        print("Invalid category choice. Please enter 0 or 1.")
        return
    
    # Set category and label array based on user choice
    category = category_choice
    label_array = labels_acorn if category_choice == 0 else labels_beetle
    
    # List and sort image files in the folder
    image_files = sorted(os.listdir(folder_path))
    
    # Iterate over image files based on numbered filenames
    for image_file in image_files:
        if image_file.lower().endswith(VALID_EXTENSIONS):
            try:
                # Extract the number from the filename
                file_number = int(''.join(filter(str.isdigit, image_file)))
                
                # Determine the label based on the number in the filename
                label_index = (file_number - 1) // 10  # Assuming 10 images per label
                if label_index < len(label_array):
                    label = label_array[label_index]
                    
                    # Process the image
                    zoi(model, os.path.join(folder_path, image_file), label, top_percentage, distance_threshold, category, resolution)
                else:
                    print(f"Label index out of range for file: {image_file}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        else:
            print(f"Skipping non-image file: {image_file}")
    
    print("Processing complete.")


if __name__ == "__main__":
    main()
