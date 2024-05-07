import os
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

def detect_outer_edge(dicom_image):
    try:
        # Check if the image is grayscale
        if len(dicom_image.shape) > 2:
            # Convert DICOM image to grayscale
            dicom_image_gray = cv2.cvtColor(dicom_image, cv2.COLOR_BGR2GRAY)
        else:
            dicom_image_gray = dicom_image
        
        # Ensure the image is of type CV_8U
        dicom_image_gray = cv2.convertScaleAbs(dicom_image_gray)
        
        # Apply Canny edge detection
        edge_image = cv2.Canny(dicom_image_gray, 100, 200)
        
        # Find contours in the binary edge image
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find the outermost contour
        outer_contour = max(contours, key=cv2.contourArea)
        
        return outer_contour
        
    except Exception as e:
        print(f"An error occurred while detecting the outer edge: {e}")
        return None

def find_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY
    else:
        return None

def convert_cm_to_pixels(distance_cm, pixel_spacing):
    # Convert centimeters to millimeters
    distance_mm = distance_cm * 10
    # Convert millimeters to pixels using pixel spacing
    distance_pixels = distance_mm / pixel_spacing
    return distance_pixels

def compute_esf(roi):
    # Compute intensity values along the x-axis (horizontal axis)
    intensity_values = np.mean(roi, axis=0)
    
    # Define the x-axis positions (upward positions of each pixel)
    x_positions = np.arange(len(intensity_values))
    
    # Interpolate the intensity values to get a smooth ESF curve
    f = interp1d(x_positions, intensity_values, kind='cubic')
    x_interp = np.linspace(0, len(intensity_values) - 1, len(intensity_values) * 4)
    esf_interp = f(x_interp)
    
    return esf_interp

def compute_lsf(esf):
    # Differentiate the ESF to obtain the LSF
    lsf = np.gradient(esf)
    
    # Zero the LSF tails by subtracting the average of the left-most part
    # of the curve from all the values of the curve
    left_part_avg = np.mean(lsf[:5])
    lsf = lsf - left_part_avg
    
    # Normalize the LSF
    lsf = lsf / np.max(lsf)
    
    return lsf

def analyze_dicom_folder(folder_path, roi_size):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            ds = pydicom.dcmread(file_path)
            dicom_image = ds.pixel_array
            pixel_spacing = ds.PixelSpacing[0]  # Assume isotropic pixel spacing
            
            # Detect outer edge of the object
            outer_contour = detect_outer_edge(dicom_image)
            
            if outer_contour is not None:
                # Create a figure with three subplots (1 row, 3 columns)
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                
                # Plot the original image
                axs[0].imshow(dicom_image, cmap='gray')
                axs[0].set_title('Original Image')
                
                # Extract contour coordinates
                x = outer_contour[:, 0, 0]
                y = outer_contour[:, 0, 1]
                
                # Plot contour over original image
                axs[0].plot(x, y, color='red', linewidth=2)
                
                # Find and plot contour center
                center = find_contour_center(outer_contour)
                if center is not None:
                    # Find the highest HU value pixel in the image
                    max_hu_pixel = np.unravel_index(np.argmax(dicom_image), dicom_image.shape)
                    
                    # Plot the highest HU value pixel in red
                    axs[0].plot(max_hu_pixel[1], max_hu_pixel[0], 'b', markersize=5)
                    
                    # Calculate the top-left corner coordinates of the ROI
                    roi_top_left = (max_hu_pixel[1] - roi_size // 2, max_hu_pixel[0] - roi_size // 2)
                    # If ROI size is odd, adjust by 1 pixel to ensure centering
                    if roi_size % 2 != 0:
                        roi_top_left = (roi_top_left[0] - 1, roi_top_left[1] - 1)
                    
                    # Draw ROI around the marked point
                    roi_rectangle = plt.Rectangle(roi_top_left, roi_size, roi_size, linewidth=1, edgecolor='red', facecolor='none')
                    axs[0].add_patch(roi_rectangle)
                    
                    # Extract pixels within the ROI
                    roi = dicom_image[roi_top_left[1]:roi_top_left[1] + roi_size, roi_top_left[0]:roi_top_left[0] + roi_size]
                    
                    # Compute ESF
                    esf = compute_esf(roi)
                    
                    # Plot ESF
                    pixel_positions = np.arange(len(esf))
                    mm_positions = pixel_positions * pixel_spacing
                    axs[1].plot(mm_positions, esf, label='ESF')
                    axs[1].set_title('Edge Spread Function (ESF)')
                    axs[1].set_xlabel('Distance (mm)')
                    axs[1].set_ylabel('HU')
                    
                    # Compute LSF
                    lsf = compute_lsf(esf)
                    
                    # Plot LSF
                    axs[2].plot(mm_positions, lsf, label='LSF')
                    axs[2].set_title('Line Spread Function (LSF)')
                    axs[2].set_xlabel('Distance (mm)')
                    axs[2].set_ylabel('Intensity')
                    
                    # Show legend for LSF plot
                    axs[2].legend()
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Show the figure
                    plt.show()
                
                
        except (pydicom.errors.InvalidDicomError, IOError) as e:
            print(f"Skipping non-DICOM file: {file_name}")
            print(f"Error: {e}")
            
# Example usage:
dicom_folder_path = r"D:\Github\MPH3013-Special-Project\Coding\Images to be processed"
roi_size_input = int(input("Enter the size of the ROI in pixels: "))
analyze_dicom_folder(dicom_folder_path, roi_size_input)
