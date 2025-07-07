from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import czifile
import nd2
import tifffile
import numpy as np
import os

def get_gpu_details():
    devices = device_lib.list_local_devices()
    for device in devices:
        if device.device_type == 'GPU':
            print(f"Device name: {device.name}")
            print(f"Device type: {device.device_type}")
            print(f"GPU model: {device.physical_device_desc}")

def list_images (directory_path):

    # Create an empty list to store all image filepaths within the dataset directory
    images = []

    # Iterate through the .czi and .nd2 files in the directory
    for file_path in directory_path.glob("*.czi"):
        images.append(str(file_path))
        
    for file_path in directory_path.glob("*.nd2"):
        images.append(str(file_path))

    return images


def read_image (image, slicing_factor_xy, slicing_factor_z):
    """Read raw image microscope files (.nd2 and .czi), apply downsampling if needed and return filename and a numpy array
    Originally intended to read multichannel 3D-stack images (ch, z, x, y), if input image is a multichannel
    2D-image the function generates a fake stack of shape (ch, 2, x, y) where both z-slices contain the same
    information. This would be transformed to the original input 2D-image when segmentation_type = '2D' """
    # Read path storing raw image and extract filename
    file_path = Path(image)
    filename = file_path.stem

    # Extract file extension
    extension = file_path.suffix

    # Read the image file (either .czi or .nd2)
    if extension == ".czi":
        # Read stack from .czi (ch, z, x, y) or (ch, x, y)
        img = czifile.imread(image)
        # Remove singleton dimensions
        img = img.squeeze()
        # Check if input image is a multichannel 3D-stack or a multichannel 2D-image
        # If multichannel 2D-image simulate a 3D-stack with 2 equal z-slices
        # I know inefficient, but do not want to change all the downstream code
        if len(img.shape) < 4:
            # Build a (ch, 2, x, y) stack
            img = np.stack([img, img], axis=1)

    elif extension == ".nd2":
        # Read stack from .nd2 (z, ch, x, y) or (ch, x, y)
        img = nd2.imread(image)
        # Check if input image is a multichannel 3D-stack or a multichannel 2D-image
        # If multichannel 2D-image simulate a 3D-stack with 2 equal z-slices
        # I know inefficient, but do not want to change all the downstream code
        if len(img.shape) < 4:
            # Build a (ch, 2, x, y) stack
            img = np.stack([img, img], axis=1)
        # This is the case of a multichannel 3D-stack (z, ch, x, y)
        else:
            # Transpose to output (ch, z, x, y)
            img = img.transpose(1, 0, 2, 3)
        
    else:
        print ("Implement new file reader")

    print(f"\n\nImage analyzed: {filename}")
    print(f"Original Array shape: {img.shape}")

    # Apply slicing trick to reduce image size (xy resolution)
    try:
        img = img[:, ::slicing_factor_z, ::slicing_factor_xy, ::slicing_factor_xy]
    except IndexError as e:
        print(f"Slicing Error: {e}")
        print(f"Slicing parameters: Slicing_XY:{slicing_factor_xy} Slicing_Z:{slicing_factor_z} ")

    # Feedback for researcher
    print(f"Compressed Array shape: {img.shape}")

    return img, filename

def maximum_intensity_projection (img):

    # Perform MIP on all channels 
    img_mip = np.max(img, axis=1)

    return  img_mip

def save_rois(viewer, directory_path, filename):

    """Code snippet to save cropped regions (ROIs) defined by labels as .tiff files"""

    # Initialize empty list to store the label name and Numpy arrays so we can loop across the different ROIs
    layer_names = []
    layer_labels = []

    if len(viewer.layers) == 1:

        print("No user-defined ROIs have been stored")

    else:

        for layer in viewer.layers:

            # Extract the label names
            label_name = layer.name
            # Ignore img_mip since it is not a user defined label
            if label_name == "img_mip":
                pass
            else:
                # Store label names
                layer_names.append(label_name)
                # Get the label data as a NumPy array to mask the image
                label = layer.data 
                layer_labels.append(label)

        # Print the defined ROIs that will be analyzed
        print(f"The following labels will be analyzed: {layer_names}")

    # Save user-defined ROIs in a ROI folder under directory_path/ROI as .tiff files
    # Subfolders for each user-defined label region
    # Store using the same filename as the input image to make things easier

    for label_name, label_array in zip(layer_names, layer_labels):

        # Perform maximum intensity projection (MIP) from the label stack
        label_mip = np.max(label_array, axis=0)

        # We will create a mask where label_mip is greater than or equal to 1
        mask = (label_mip >= 1).astype(np.uint8)

        # Create ROI directory if it does not exist
        try:
            os.makedirs(directory_path / "ROIs" / label_name)
        except FileExistsError:
            pass

        # Construct path to store
        roi_path = directory_path / "ROIs" / label_name / f"{filename}.tiff"

        # Save mask (binary image)
        tifffile.imwrite(roi_path, mask)

def plot_segmentation(plots):
    """Takes as an input a list of dictionaries containing filename, roi_name and numpy arrays for input image and resulting marker+ ROI"""
    
    for plot in plots:

        # Dynamically set the number of subplots based on the number of markers
        subplots_nr = len(plot["markers"]) * 2  # Total number of subplots

        plt.figure(figsize=(50, 25))  # Slightly larger figure for better readability

        position = 1

        # Suptitle for the entire figure
        plt.suptitle(f"Filename: {plot['filename']}    ROI: {plot['roi']}", 
                     fontsize=50, fontweight="bold", y=1.05)  # y > 1 moves it up

        for marker in plot["markers"]:
            img1 = marker[1]  # First image (grayscale)
            img2 = marker[2]  # Second image (mask)
            
            # Stretch contrast for the first image only
            vmin1, vmax1 = np.min(img1), np.max(img1)

            plt.subplot(1, subplots_nr, position)
            plt.imshow(img1, cmap="gray", vmin=vmin1, vmax=vmax1)  # Adjust contrast
            plt.title(f'Input {marker[0]} MIP Image', fontsize=50)
            plt.axis("off")
            position += 1

            plt.subplot(1, subplots_nr, position)
            plt.imshow(img2, cmap="viridis")  # No contrast adjustment for mask
            plt.title(f"{marker[0]}+_ROI", fontsize=50)
            plt.axis("off")
            position += 1

    plt.tight_layout()  # Adjusts layout to reduce space
    plt.subplots_adjust(top=0.95)  # Moves subplots down to create more space for suptitle

    plt.show()