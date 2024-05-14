import os
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def detect_outer_edge(dicom_image):
    try:
        if len(dicom_image.shape) > 2:
            dicom_image_gray = cv2.cvtColor(dicom_image, cv2.COLOR_BGR2GRAY)
        else:
            dicom_image_gray = dicom_image
        
        dicom_image_gray = cv2.convertScaleAbs(dicom_image_gray)
        
        edge_image = cv2.Canny(dicom_image_gray, 100, 200)
        
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
    lsf = np.sum(roi_ct_numbers, axis=1)

    # Subtract background values
    background = np.min(lsf)
    lsf -= background

    # Interpolate the LSF using CubicSpline
    interpolated_positions = np.linspace(0, len(lsf) - 1, num=100*len(lsf))
    cubic_spline = CubicSpline(np.arange(len(lsf)), lsf)
    interpolated_lsf = cubic_spline(interpolated_positions)

    # Normalize the LSF
    smallest_values = np.partition(interpolated_lsf, 5)[:5]
    average_smallest_values = np.mean(smallest_values)
    normalized_lsf = (interpolated_lsf - average_smallest_values) / (np.max(interpolated_lsf) - average_smallest_values)
    
    return normalized_lsf

def calculate_mtf(normalized_lsf):
    mtf = np.abs(np.fft.fft(normalized_lsf))
    mtf_normalized = mtf / mtf[0]  # Normalize by the DC component
    
    return mtf_normalized

def calculate_spatial_frequency(pixel_spacing, num_points):
    freq = np.fft.fftfreq(num_points, d=pixel_spacing)
    return freq[:num_points // 2]  # Only take the positive frequencies

def plotting(dicom_image, outer_contour, roi_top_left, roi_size, normalized_lsf, mtf_normalized, axs, pixel_spacing, spatial_frequency):
    axs[0].imshow(dicom_image, cmap='gray')
    x = outer_contour[:, 0, 0]
    y = outer_contour[:, 0, 1]
    axs[0].plot(x, y, color='red', linewidth=2)
    roi_rectangle = plt.Rectangle(roi_top_left, roi_size, roi_size, linewidth=1, edgecolor='red', facecolor='none')
    axs[0].add_patch(roi_rectangle)
    axs[0].set_title('Original Image with ROI')
    
    x_mm = np.arange(len(normalized_lsf)) * (pixel_spacing / 10)

    axs[1].plot(x_mm, normalized_lsf)
    axs[1].set_xlabel('Position (mm)')
    axs[1].set_ylabel('Normalized CT Numbers')
    axs[1].set_title('Line Spread Function (LSF)')
    axs[1].grid(True)
    
    spatial_frequency_positive = spatial_frequency[:len(spatial_frequency)//2]
    axs[2].plot(spatial_frequency_positive, mtf_normalized[:len(spatial_frequency_positive)])
    axs[2].set_xlabel('Spatial Frequency (cycles/mm)')
    axs[2].set_ylabel('MTF')
    axs[2].set_title('Modulation Transfer Function (MTF)')
    axs[2].grid(True)

def position_and_draw_roi(dicom_image, outer_contour, center, roi_size, pixel_spacing):
    try:
        # Calculate the top left corner of the larger ROI (25x25)
        roi_large_top_left = (center[0] - int(1 / pixel_spacing), center[1] + int(60 / pixel_spacing))

        # Extract the larger ROI (25x25)
        roi_large_ct_numbers = dicom_image[roi_large_top_left[1]:roi_large_top_left[1] + 25, 
                                           roi_large_top_left[0]:roi_large_top_left[0] + 25]

        # Calculate the top left corner of the smaller ROI around the highest point
        max_hu_pixel_roi = np.unravel_index(np.argmax(roi_large_ct_numbers), roi_large_ct_numbers.shape)
        roi_top_left = (roi_large_top_left[0] + max_hu_pixel_roi[1] - roi_size // 2,
                        roi_large_top_left[1] + max_hu_pixel_roi[0] - roi_size // 2)

        if roi_size % 2 != 0:
            roi_top_left = (roi_top_left[0] - 1, roi_top_left[1] - 1)

        # Extract the smaller ROI (given size) around the highest point
        roi_ct_numbers = dicom_image[roi_top_left[1]:roi_top_left[1] + roi_size, 
                                     roi_top_left[0]:roi_top_left[0] + roi_size]

        return roi_ct_numbers, roi_top_left

    except Exception as e:
        print(f"An error occurred while positioning and drawing ROI: {e}")
        return None, None, None

def analyze_dicom_folder(folder_path, roi_size):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            ds = pydicom.dcmread(file_path)
            dicom_image = ds.pixel_array
            pixel_spacing = ds.PixelSpacing[0]
            
            outer_contour = detect_outer_edge(dicom_image)
            
            if outer_contour is not None:
                fig, axs = plt.subplots(1, 3)
                
                center = find_contour_center(outer_contour)
                if center is not None:
                    roi_ct_numbers, roi_top_left = position_and_draw_roi(dicom_image, outer_contour, center, roi_size, pixel_spacing)

                    normalized_lsf = calculate_lsf(roi_ct_numbers)
                    
                    mtf_normalized = calculate_mtf(normalized_lsf)
                    
                    spatial_frequency = calculate_spatial_frequency(pixel_spacing / 10, len(normalized_lsf))
                    
                    plotting(dicom_image, outer_contour, roi_top_left, roi_size, normalized_lsf, mtf_normalized, axs, pixel_spacing, spatial_frequency)
                    
                    plt.tight_layout()
                    plt.show()
                
        except (pydicom.errors.InvalidDicomError, IOError) as e:
            print(f"Skipping non-DICOM file: {file_name}")
            print(f"Error: {e}")

# Example usage:
dicom_folder_path = r"C:\Users\farru\Documents\Github\MPH3013-Special-Project\Coding\Images to be processed"
roi_size_input = int(input("Enter the size of the smaller ROI in pixels: "))  # For the given size ROI
analyze_dicom_folder(dicom_folder_path, roi_size_input)
