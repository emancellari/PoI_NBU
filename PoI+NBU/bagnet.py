
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision import transforms
from PIL import Image, ImageDraw
from sklearn.cluster import DBSCAN
from bagnet_utils  import generate_heatmap_pytorch, plot_heatmap
import pytorchnet
import os
import shutil
from torch.nn.functional import softmax
import time
import cv2


def zoi(model, image_path, label, top_percentage, distance_threshold, category,resolution):
    start_time_program = time.time()


    # Function to get the gradients of the input image
    def get_gradients(model, input_tensor, label):
        input_var = torch.autograd.Variable(input_tensor, requires_grad=True)
        output = model(input_var)
        loss = output[0, label]  # Assuming single-label classification
        loss.backward()
        return input_var.grad.data


    # Function to print positions of top x% gradients
    def get_top_gradients_positions(gradients, top_percentage):
        flat_gradients = gradients.flatten()
        threshold_value = np.percentile(np.abs(flat_gradients), 100 - top_percentage)
        high_gradient_positions = np.column_stack(np.where(np.abs(gradients) >= threshold_value))

        content = f""
        pixel_count = 0

        for pos in high_gradient_positions:
            pixel_position = f"{pos[2]}, {pos[3]}\n"
            content += pixel_position
            pixel_count += 1

        #content += f"Total number of pixels with high gradients: {pixel_count}"
        return content

    def extract_top_gradients_positions(gradients, top_percentage):
        flat_gradients = gradients.flatten()
        threshold_value = np.percentile(np.abs(flat_gradients), 100 - top_percentage)
        high_gradient_positions = np.column_stack(np.where(np.abs(gradients) >= threshold_value))

        gradient_positions = []

        for pos in high_gradient_positions:
            gradient_positions.append((pos[2], pos[3]))

        return gradient_positions
    def extract_top_gradients_positions_small(gradients, top_percentage):
        flat_gradients = gradients.flatten()
        threshold_value = np.percentile(np.abs(flat_gradients), 100 - top_percentage)
        high_gradient_positions = np.column_stack(np.where(np.abs(gradients) >= threshold_value))

        gradient_positions = []

        for pos in high_gradient_positions:
            gradient_positions.append((pos[3], pos[2]))

        return gradient_positions

    def rescale_positions(gradient_positions, original_width, original_height, resized_width=224, resized_height=224):
        # Calculate scaling factors
        width_scale = original_width / resized_width
        height_scale = original_height / resized_height

        # Rescale positions to original image size
        rescaled_positions = []
        for x, y in gradient_positions:
            x_rescaled = int(y * width_scale)
            y_rescaled = int(x * height_scale)
            rescaled_positions.append((x_rescaled, y_rescaled))

        return rescaled_positions

    def visualize_pixels(positions, image_size, dot_size=1):
        """
        Visualize pixels as dots on a white background with increased dot density.

        Parameters:
        - positions: List of tuples containing (x, y) positions of pixels.
        - image_size: Tuple containing the width and height of the image.
        - dot_size: Size of the dots.

        Returns:
        - None
        """
        # Create a white canvas
        canvas = np.ones(image_size, dtype=np.uint8) * 255

        # Plot pixels as dots
        for x, y in positions:
            canvas[max(0, y - dot_size):min(image_size[1], y + dot_size + 1), max(0, x - dot_size):min(image_size[0], x + dot_size + 1)] = 0

        # Display the canvas
        plt.imshow(canvas, cmap='gray')
        plt.axis('off')
        #plt.show()

    def visualize_rescaled_pixels(positions, original_image):
        """
        Visualize rescaled pixels as circles or bullets in the original image.

        Parameters:
        - positions: List of tuples containing (x, y) positions of pixels.
        - original_image: Original image as a NumPy array.

        Returns:
        - None
        """
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Show the original image
        ax.imshow(original_image)

        # Plot circles at rescaled pixel positions
        for x, y in positions:
            ax.plot(x, y, 'ro', markersize=3)  # Adjust markersize as needed

        # Turn off axis
        ax.axis('off')

        # Show the plot
        #plt.show()
        plt.close()


        return fig
    def visualize_pixel_regions(positions, original_image, distance_threshold, output_file):
        """
        Visualize pixels that are near to each other with a specific distance and identify those regions
        by creating rectangles as borders.

        Parameters:
        - positions: List of tuples containing (x, y) positions of pixels.
        - original_image: Original image as a NumPy array.
        - distance_threshold: Maximum distance between two pixels to be considered as part of the same region.
        - output_file: Path to the output file to save the rectangle positions.

        Returns:
        - List of tuples containing the positions of the rectangles.
        """
        # Convert positions to NumPy array
        points = np.array(positions)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=distance_threshold, min_samples=1)
        clusters = dbscan.fit_predict(points)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Show the original image
        ax.imshow(original_image)

        # Initialize list to store rectangle positions
        rectangle_positions = []


        # Plot rectangles around pixel clusters
        for cluster_id in np.unique(clusters):
            cluster_points = points[clusters == cluster_id]
            min_x, min_y = np.min(cluster_points, axis=0)
            max_x, max_y = np.max(cluster_points, axis=0)
            width = max_x - min_x
            height = max_y - min_y
            rect = Rectangle((min_x, min_y),width, height, fill=None, edgecolor='r', linewidth=2)
            ax.add_patch(rect)

            # Add rectangle vertices to the list
            rectangle_positions.append(((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)))


        # Turn off axis
        ax.axis('off')

        # Save rectangle positions to a text file
        with open(output_file, 'w') as f:
            for rect_pos in rectangle_positions:
                f.write(','.join([f'{x},{y}' for x, y in rect_pos]) + '\n')

        # Show the plot
        #plt.show()
        plt.close()

        return fig, rectangle_positions


    def show_top_gradients_heatmap(gradients, top_percentage):
        # Take the magnitude of gradients across color channels
        magnitude_gradients = np.linalg.norm(gradients, axis=1)

        flat_gradients = magnitude_gradients.flatten()
        threshold_value = np.percentile(np.abs(flat_gradients), 100 - top_percentage)
        high_gradient_positions = np.column_stack(np.where(np.abs(magnitude_gradients) >= threshold_value))

        # Create a binary mask for the top x% magnitude gradients
        top_gradients_mask = np.zeros_like(magnitude_gradients[0])  # Assuming gradients has shape (1, 3, height, width)
        for pos in high_gradient_positions:
            top_gradients_mask[tuple(pos[-2:])] = 1  # Use the last two values as height and width indices

        # Show the heatmap
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f'Top {top_percentage}% Gradients Heatmap')
        cax = ax.imshow(top_gradients_mask, cmap='hot', alpha=0.5)
        plt.colorbar(cax)
        plt.axis('off')
        #plt.show()
        plt.close()


    def save_txt_file(data, file_path):
        # Convert the list to a set to remove duplicates
        unique_data = set(data)
        with open(file_path, 'w') as f:
            for item in unique_data:
                f.write(str(item) + "\n")

    def save_txt_file_1(content, file_path):
        with open(file_path, 'w') as file:
            file.write(content)

    def save_heatmap_figure(gradients, top_percentage, save_path):
        magnitude_gradients = np.linalg.norm(gradients, axis=1)
        flat_gradients = magnitude_gradients.flatten()
        threshold_value = np.percentile(np.abs(flat_gradients), 100 - top_percentage)
        high_gradient_positions = np.column_stack(np.where(np.abs(magnitude_gradients) >= threshold_value))

        top_gradients_mask = np.zeros_like(magnitude_gradients[0])
        for pos in high_gradient_positions:
            top_gradients_mask[tuple(pos[-2:])] = 1

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f'Top {top_percentage}% Gradients Heatmap')
        cax = ax.imshow(top_gradients_mask, cmap='hot', alpha=0.5)
        plt.colorbar(cax)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def extract_numbers(input_file, output_file):
        # Initialize an empty list to store extracted numbers
        extracted_numbers_list = []

        # Open the input file
        with open(input_file, 'r') as f:
            # Read lines from the input file
            lines = f.readlines()

        # Iterate over each line in the input file
        for line in lines:
            # Split the line into individual numbers
            numbers = line.strip().split(',')
            # Extract the desired numbers and append them to the list
            extracted_numbers_list.append([int(numbers[0]), int(numbers[2]), int(numbers[3]), int(numbers[5])])

        # Convert the list to a NumPy array
        extracted_numbers_array = np.array(extracted_numbers_list)

        # Save the NumPy array to a file
        np.save(output_file, extracted_numbers_array)

    def scale_rectangles(rect_pos, old_width, old_height, new_width, new_height,output_file):
        # Calculate scaling factors
        width_scale = new_width / old_width
        height_scale = new_height / old_height

        # Scale each rectangle position
        scaled_rects = []
        for rect in rect_pos:
            # Unpack the coordinates from the tuple
            x1, y1 = rect[0]
            x2, y2 = rect[2]

            # Scale each coordinate individually
            scaled_x1 = int(x1 * width_scale)
            scaled_y1 = int(y1 * height_scale)
            scaled_x2 = int(x2 * width_scale)
            scaled_y2 = int(y2 * height_scale)

            # Append the scaled rectangle to the list
            scaled_rects.append(
                ((scaled_x1, scaled_y1), (scaled_x2, scaled_y1), (scaled_x2, scaled_y2), (scaled_x1, scaled_y2)))

            # Save scaled rectangles to output file
        with open(output_file, 'w') as f:
            for rect in scaled_rects:
                # Write the coordinates of each corner of the rectangle to the file
                f.write(','.join([f'{x},{y}' for x, y in rect]) + '\n')

        return scaled_rects

    def draw_rectangles(image, scaled_rect_pos, output_path):
        # Create a drawing object
        draw = ImageDraw.Draw(image)

        # Draw red rectangles
        for rect in scaled_rect_pos:
            # Unpack the coordinates of the rectangle corners
            x1, y1 = rect[0]
            x2, y2 = rect[2]

            # Define rectangle coordinates (top-left and bottom-right)
            rectangle = [(x1, y1), (x2, y2)]

            # Draw the rectangle
            draw.rectangle(rectangle, outline="red", width=2)

        # Save the image with rectangles drawn
        image.save(output_path)
        #image.show()





    start_time_load_model = time.time()
    # Load the Bagnet-9-17-33 model

    pytorch_model = model
    pytorch_model.eval()

    end_time_load_model = time.time()
    load_model_time = end_time_load_model - start_time_load_model

    # Load an image from your local path

    image_path = image_path

    original_image = Image.open(image_path).convert('RGB')


    #Preprocess the sample image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #
    sample = preprocess(original_image)
    sample = sample.unsqueeze(0)  # Add batch dimension

    preproces = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Apply preprocessing
    preprocessed_image = preproces(original_image)

    # Convert tensor to PIL Image
    preprocessed_image_pil = transforms.ToPILImage()(preprocessed_image)

    # Save the preprocessed image as JPEG
    preprocessed_image_pil.save(f"preprocessed_image_{category}.jpg", "JPEG")


    # Get the model's predictions (logits)
    logits = pytorch_model(sample)

    label=label
    category=category
    #print(logits)
    print(label)

    #step2 time

    start_step2_time = time.time()
    # Load the Bagnet-9-17-33 model

    pytorch_model = model
    pytorch_model.eval()

    end_time_step2_time = time.time()
    step2_time = end_time_step2_time - start_step2_time

    # Get the gradients of the input with respect to the predicted class
    gradients = get_gradients(pytorch_model, sample, label)

    #print(gradients)

    start_step3_time=time.time()
    top_percentage=top_percentage


    gradient_positions_content = get_top_gradients_positions(gradients.numpy(), top_percentage)

    end_time_step3_time = time.time()
    step3_time = end_time_step3_time - start_step3_time

    # Save gradient positions to a txt file
    gradient_positions_file_path = f"gradient_positions_{category}_{top_percentage}.txt"
    save_txt_file_1(gradient_positions_content, gradient_positions_file_path)



    # Assuming `gradients` is the gradients array you want to visualize
    show_top_gradients_heatmap(gradients.numpy(), top_percentage)
    # saving top percentage image
    heatmap_figure_save_path = "heatmap_top.png"
    save_heatmap_figure(gradients.numpy(), top_percentage, heatmap_figure_save_path)

    gradient_positions_content = extract_top_gradients_positions(gradients.numpy(), top_percentage)
    gradient_positions_content_small = extract_top_gradients_positions_small(gradients.numpy(), top_percentage)
    #print(gradient_positions_content)

    original_width, original_height = original_image.size

    print(original_width)
    print(original_height)

    print("=========================")

    rescaled_positions_original_size = rescale_positions(gradient_positions_content, original_width, original_height)

    #print(rescaled_positions_original_size)

    rescaled_gradient_positions_file_path = f"rescaled_gradient_positions_{category}.txt"
    save_txt_file(rescaled_positions_original_size, rescaled_gradient_positions_file_path)

    #visualize_pixels(rescaled_positions_original_size, (original_width, original_height),dot_size=3)

    pixels_figure=visualize_rescaled_pixels(rescaled_positions_original_size, original_image)


    pixels_figure.savefig(f'rescaled_pixels_figure_{category}.png')

    start_step4_time=time.time()

    distance_threshold=distance_threshold

    output_file_hr = f"rectangle_positions_hr_{category}.txt"
    output_file_lr = f"rectangle_positions_lr_{category}.txt"
    #zoi_figures, rectangle_positions=visualize_pixel_regions(rescaled_positions_original_size, original_image, distance_threshold,output_file=output_file_hr)
    # image_original=original_image.resize((224,224))
    zoi_figures2,rect_pos=visualize_pixel_regions(gradient_positions_content_small, preprocessed_image_pil, distance_threshold,output_file=output_file_lr)
    #zoi_figures.savefig('zoi_figure_hr.png')

    end_time_step4_time=time.time()

    step4_time=end_time_step4_time-start_step4_time

    zoi_figures2.savefig(f'zoi_figure_lr_{category}.png')
    # Generate heatmap for the original image

    old_width, old_height = 224, 224  # Original image dimensions


    scaled_rect_pos=scale_rectangles(rect_pos, old_width, old_height, original_width, original_height, output_file_hr)

    zoi_figure_hr = f"zoi_figure_hr_{category}.jpg"



    draw_rectangles(original_image, scaled_rect_pos, zoi_figure_hr)



    # Example usage:

    resolution=resolution
    if resolution==1: #hr
        input_file = output_file_hr
        if category== 1:
            rectangles_file = "extracted_numbers_ca_hr.npy"
        else:
            rectangles_file = "extracted_numbers_ct_hr.npy"

    else:
        input_file = output_file_lr
        if category == 1:
            rectangles_file = "extracted_numbers_ca_lr.npy"
        else:
            rectangles_file = "extracted_numbers_ct_lr.npy"

    start_step5_time=time.time()
    extract_numbers(input_file, rectangles_file)

    end_time_step5_time = time.time()

    step5_time = end_time_step5_time - start_step5_time
    test=np.load(rectangles_file)
    #print(test)



    heatmap_original = generate_heatmap_pytorch(pytorch_model, sample, label, 33)

    # Convert the original tensor to a NumPy array
    original_image_np = np.array(original_image)

    # Save original image
    original_image_save_path = "original_image.png"
    output_path = "output_image.jpg"  # Specify the path where you want to save the image
    original_image.save(output_path)

    plt.imshow(original_image_np / 255.)
    plt.axis('off')
    plt.savefig(original_image_save_path)

    plt.close()






    # Save heatmap for the original image
    heatmap_original_save_path = f"heatmap_original_{category}.png"
    plot_heatmap(heatmap_original, original_image_np, None, dilation=0.5, percentile=99, alpha=.25)
    plt.axis('off')

    # Adjust figure size
    plt.gcf().set_size_inches(original_image.size[0] / 100, original_image.size[1] / 100)

    # Save the figure without white background
    plt.savefig(heatmap_original_save_path, bbox_inches='tight', pad_inches=0,dpi='figure')

    plt.close()





    # Move the saved files to a 'figures' folder
    figures_folder = f"Figures/{image_path}"
    os.makedirs(figures_folder, exist_ok=True)


    shutil.move(gradient_positions_file_path, os.path.join(figures_folder, gradient_positions_file_path))
   # shutil.move(gradient_file_path, os.path.join(figures_folder, gradient_file_path))
    shutil.move(rescaled_gradient_positions_file_path, os.path.join(figures_folder, rescaled_gradient_positions_file_path))
    shutil.move(original_image_save_path, os.path.join(figures_folder, original_image_save_path))
    shutil.move(heatmap_original_save_path, os.path.join(figures_folder, heatmap_original_save_path))
    shutil.move(heatmap_figure_save_path, os.path.join(figures_folder, heatmap_figure_save_path))
    shutil.move(f'rescaled_pixels_figure_{category}.png', os.path.join(figures_folder, f'rescaled_pixels_figure_{category}.png'))
    shutil.move(zoi_figure_hr, os.path.join(figures_folder, zoi_figure_hr))
    shutil.move(f'zoi_figure_lr_{category}.png', os.path.join(figures_folder, f'zoi_figure_lr_{category}.png'))
    shutil.move(rectangles_file, os.path.join(figures_folder, rectangles_file))

    end_time_program = time.time()
    total_program_time = end_time_program - start_time_program

    # Save the pixel prediction time and total program time to a text file
    time_file_path = "exec_time_ct.txt"
    with open(time_file_path, 'w') as time_file:
        time_file.write(f"Total Program Execution Time: {total_program_time} seconds\n")
        time_file.write(f"Load Model Execution Time: {load_model_time} seconds\n")
        time_file.write(f"Step 2 Execution Time: {step2_time} seconds\n")
        time_file.write(f"Step 3 Execution Time: {step3_time} seconds\n")
        time_file.write(f"Step 4 Execution Time: {step4_time} seconds\n")
        time_file.write(f"Step 5 Execution Time: {step5_time} seconds\n")

    shutil.move(time_file_path, os.path.join(figures_folder, time_file_path))


    # Display the figures folder path
    print(f"Saved figures in: {figures_folder}")

