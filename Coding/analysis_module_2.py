import os
import cv2
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
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

def convert_cm_to_pixels(distance_cm, pixel_spacing):
    # Convert centimeters to millimeters
    distance_mm = distance_cm * 10
    # Convert millimeters to pixels using pixel spacing
    distance_pixels = distance_mm / pixel_spacing
    return distance_pixels

from scipy.interpolate import interp1d

def compute_esf(roi):
    # Compute intensity values along the vertical axis
    intensity_values = np.mean(roi, axis=0)
    
    # Smooth the intensity values to reduce noise
    smoothed_intensity = convolve(intensity_values, np.ones(5)/5, mode='same')
    
    # Compute the derivative of the intensity values
    derivative = np.gradient(smoothed_intensity)
    
    # Calculate the cumulative sum of the derivative to obtain the ESF
    esf = np.cumsum(derivative)
    
    # Interpolate the ESF to get 4 points per pixel
    x_orig = np.arange(len(esf))
    f = interp1d(x_orig, esf, kind='cubic')
    x_interp = np.linspace(0, len(esf) - 1, len(esf) * 4)
    esf_interp = f(x_interp)
    
    return esf_interp

def compute_lsf(roi):
    # Compute intensity values along the vertical axis
    intensity_values = np.mean(roi, axis=0)
    
    # Smooth the intensity values to reduce noise
    smoothed_intensity = convolve(intensity_values, np.ones(5)/5, mode='same')
    
    # Compute the derivative of the intensity values
    derivative = np.gradient(smoothed_intensity)
    
    # Differentiate the ESF to obtain the LSF
    lsf = np.gradient(derivative)
    
    # Interpolate the LSF to get 4 points per pixel
    x_orig = np.arange(len(lsf))
    f = interp1d(x_orig, lsf, kind='cubic')
    x_interp = np.linspace(0, len(lsf) - 1, len(lsf) * 4)
    lsf_interp = f(x_interp)
    
    return lsf_interp

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
                # Plot the original image
                plt.imshow(dicom_image, cmap='gray')
                
                # Extract contour coordinates
                x = outer_contour[:, 0, 0]
                y = outer_contour[:, 0, 1]
                
                # Plot contour over original image
                plt.plot(x, y, color='red', linewidth=2)
                
                # Find and plot contour center
                center = find_contour_center(outer_contour)
                if center is not None:
                    # Calculate distance_pixels
                    distance_pixels = convert_cm_to_pixels(6, pixel_spacing)
                    
                    # Find the highest HU value pixel in the image
                    max_hu_pixel = np.unravel_index(np.argmax(dicom_image), dicom_image.shape)
                    
                    # Plot the highest HU value pixel in red
                    plt.plot(max_hu_pixel[1], max_hu_pixel[0], 'ro', markersize=5)
                    
                    # Draw ROI around the marked point
                    roi_top_left = (max_hu_pixel[1] - roi_size // 2, max_hu_pixel[0] - roi_size // 2)
                    roi_rectangle = plt.Rectangle(roi_top_left, roi_size, roi_size, linewidth=1, edgecolor='red', facecolor='none')
                    plt.gca().add_patch(roi_rectangle)
                    
                    # Extract pixels within the ROI
                    roi = dicom_image[roi_top_left[1]:roi_top_left[1] + roi_size, roi_top_left[0]:roi_top_left[0] + roi_size]
                    
                    # Compute ESF
                    esf = compute_esf(roi)
                    
                    # Plot ESF
                    pixel_positions = np.arange(len(esf))
                    mm_positions = pixel_positions * pixel_spacing
                    plt.figure()
                    plt.plot(mm_positions, esf, label='ESF')
                    
                    # Compute LSF
                    lsf = compute_lsf(roi)
                    
                    # Plot LSF
                    plt.plot(mm_positions, lsf, label='LSF')
                    
                    plt.title('Edge Spread Function (ESF) and Line Spread Function (LSF)')
                    plt.xlabel('Distance (mm)')
                    plt.ylabel('Intensity')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                
        except (pydicom.errors.InvalidDicomError, IOError) as e:
            print(f"Skipping non-DICOM file: {file_name}")
            print(f"Error: {e}")
            
# Example usage:
dicom_folder_path = r"C:\Users\farru\Documents\Github\MPH3013-Special-Project\Coding\Images to be processed"
roi_size_input = int(input("Enter the size of the ROI in pixels: "))
analyze_dicom_folder(dicom_folder_path, roi_size_input)


# Example usage:
dicom_folder_path = r"C:\Users\farru\Documents\Github\MPH3013-Special-Project\Coding\Images to be processed"
roi_size_input = int(input("Enter the size of the ROI in pixels: "))
analyze_dicom_folder(dicom_folder_path, roi_size_input)
