import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def detect_phantom_central_pixel_and_edge(dicom_file_path):
    # Read DICOM file
    dicom = pydicom.dcmread(dicom_file_path)
    
    # Extract pixel array
    image = dicom.pixel_array
    
    # Convert to uint8 (OpenCV requires uint8 for edge detection)
    image_uint8 = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_uint8, (5, 5), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate the centroid of the largest contour
    M = cv2.moments(largest_contour)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    
    return (centroid_x, centroid_y), largest_contour

def convert_physical_to_pixel_coordinates(physical_coordinates, pixel_spacing):
    # Convert physical coordinates to pixel coordinates
    pixel_coordinates = (physical_coordinates[0] / pixel_spacing[0], physical_coordinates[1] / pixel_spacing[1])
    return pixel_coordinates

# Example DICOM file path
dicom_file_path = r'D:\Github\MPH3013-Special-Project\Coding\Images to be processed\PHYSICS_CATPHAN.CT.HEAD_HEADROUTINESEQ_(ADULT).0006.0003.2023.02.08.11.59.27.521533.6462115.IMA'

# Read DICOM file
dicom = pydicom.dcmread(dicom_file_path)

# Extract pixel array
image = dicom.pixel_array

# Detect central pixel of the phantom and its edge
central_pixel, edge_contour = detect_phantom_central_pixel_and_edge(dicom_file_path)

# Extract pixel spacing information
pixel_spacing = dicom.PixelSpacing

# Calculate physical coordinates 6cm below the central pixel
physical_coordinates_below_6cm = (central_pixel[0] * pixel_spacing[0], central_pixel[1] * pixel_spacing[1] + 60)

# Convert physical coordinates to pixel coordinates
below_6cm_pixel = convert_physical_to_pixel_coordinates(physical_coordinates_below_6cm, pixel_spacing)

# Plot the original image with marked regions
plt.imshow(image, cmap='gray')

# Find the pixel with the highest HU value
max_hu_pixel = np.unravel_index(np.argmax(image), image.shape)

# Draw square of 5x5 centered on the marked pixel using matplotlib.patches.Rectangle
square_size = 5
square_half_size = square_size/2  # Adjust the size if needed
square_corner = (max_hu_pixel[1] - square_half_size, max_hu_pixel[0] - square_half_size)

rect = patches.Rectangle((square_corner[0], square_corner[1]), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')

plt.gca().add_patch(rect)

plt.scatter(max_hu_pixel[1], max_hu_pixel[0], c='r', label='Highest HU Pixel')  # Note: numpy indices are row, column
plt.legend()
plt.title('Original Image with Marked Regions, Edge, and Highest HU Pixel')
plt.show()
