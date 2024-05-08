import os
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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

def calculate_lsf(roi_ct_numbers):
    # Calculate the Line Spread Function (LSF)
    lsf = np.sum(roi_ct_numbers, axis=1)  # Sum along rows to simulate the LSF
    
    # Interpolate LSF values for a smooth curve
    interpolated_positions = np.linspace(0, len(lsf) - 1, num=10*len(lsf))  # Interpolate 10x more points
    interpolated_lsf = interp1d(np.arange(len(lsf)), lsf, kind='cubic')(interpolated_positions)
    
    # Normalize the LSF
    normalized_lsf = (interpolated_lsf - np.min(interpolated_lsf)) / (np.max(interpolated_lsf) - np.min(interpolated_lsf))
    
    return normalized_lsf

def calculate_mtf(normalized_lsf, pixel_spacing, image_size):
    # Calculate the Modulation Transfer Function (MTF) using FFT
    mtf = np.abs(np.fft.fft(normalized_lsf))
    mtf_normalized = mtf / mtf.max()  # Normalize MTF
    
    # Calculate spatial frequencies
    freq_x, freq_y = calculate_spatial_frequency(pixel_spacing, image_size)
    freq = np.sqrt(freq_x**2 + freq_y**2)
    
    return freq, mtf_normalized

def calculate_spatial_frequency(pixel_spacing, image_size):
    # Calculate spatial frequency along x and y directions
    freq_x = np.fft.fftfreq(image_size[1], pixel_spacing)
    freq_y = np.fft.fftfreq(image_size[0], pixel_spacing)
    return freq_x, freq_y


def plotting(dicom_image, outer_contour, roi_top_left, roi_size, normalized_lsf, mtf_normalized, axs, pixel_spacing):
    # Plot the original image with the outer contour and ROI
    axs[0].imshow(dicom_image, cmap='gray')
    x = outer_contour[:, 0, 0]
    y = outer_contour[:, 0, 1]
    axs[0].plot(x, y, color='red', linewidth=2)
    roi_rectangle = plt.Rectangle(roi_top_left, roi_size, roi_size, linewidth=1, edgecolor='red', facecolor='none')
    axs[0].add_patch(roi_rectangle)
    axs[0].set_title('Original Image with ROI')
    
    # Calculate x-axis in mm
    x_mm = np.arange(len(normalized_lsf)) * pixel_spacing
    
    # Plot the normalized Line Spread Function (LSF)
    axs[1].plot(x_mm, normalized_lsf)
    axs[1].set_xlabel('Position (mm)')
    axs[1].set_ylabel('Normalized CT Numbers')
    axs[1].set_title('Line Spread Function (LSF)')
    axs[1].grid(True)
    
    # Plot the MTF
    axs[2].plot(mtf_normalized)
    axs[2].set_xlabel('Spatial Frequency (cycles/pixel)')
    axs[2].set_ylabel('MTF')
    axs[2].set_title('Modulation Transfer Function (MTF)')
    axs[2].grid(True)


def analyze_dicom_folder(folder_path, roi_size):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            ds = pydicom.dcmread(file_path)
            dicom_image = ds.pixel_array
            pixel_spacing = ds.PixelSpacing[0]  # Assume isotropic pixel spacing
            image_size = dicom_image.shape
            
            # Detect outer edge of the object
            outer_contour = detect_outer_edge(dicom_image)
            
            if outer_contour is not None:
                # Create a figure with four subplots (1 row, 3 columns)
                fig, axs = plt.subplots(1, 3)
                
                # Extract contour coordinates
                x = outer_contour[:, 0, 0]
                y = outer_contour[:, 0, 1]
                
                # Find contour center
                center = find_contour_center(outer_contour)
                if center is not None:
                    # Find the highest HU value pixel in the image
                    max_hu_pixel = np.unravel_index(np.argmax(dicom_image), dicom_image.shape)
                    
                    # Calculate the top-left corner coordinates of the ROI
                    roi_top_left = (max_hu_pixel[1] - roi_size // 2, max_hu_pixel[0] - roi_size // 2)
                    # If ROI size is odd, adjust by 1 pixel to ensure centering
                    if roi_size % 2 != 0:
                        roi_top_left = (roi_top_left[0] - 1, roi_top_left[1] - 1)
                    
                    # Extract CT numbers within the automatically drawn ROI
                    roi_ct_numbers = ds.pixel_array[roi_top_left[1]:roi_top_left[1] + roi_size, roi_top_left[0]:roi_top_left[0] + roi_size]
                    
                    # Calculate LSF
                    normalized_lsf = calculate_lsf(roi_ct_numbers)
                    
                    # Calculate MTF
                    freq, mtf_normalized = calculate_mtf(normalized_lsf, pixel_spacing, image_size)

                    
                    # Plotting
                    plotting(dicom_image, outer_contour, roi_top_left, roi_size, normalized_lsf, mtf_normalized, axs, pixel_spacing)
                    
                    # Show the figure
                    plt.tight_layout()
                    plt.show()
                
        except (pydicom.errors.InvalidDicomError, IOError) as e:
            print(f"Skipping non-DICOM file: {file_name}")
            print(f"Error: {e}")

# Example usage:
dicom_folder_path = r"D:\Github\MPH3013-Special-Project\Coding\Images to be processed"
roi_size_input = int(input("Enter the size of the ROI in pixels: "))
analyze_dicom_folder(dicom_folder_path, roi_size_input)
