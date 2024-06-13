import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import cv2
from scipy.interpolate import PchipInterpolator, CubicSpline

def load_dicom_image(file_path):
    """ Load DICOM image and return pixel array """
    ds = pydicom.dcmread(file_path)
    return ds.pixel_array.astype(np.float64), ds

def find_max_hu_value(image):
    """ Find the maximum HU value in the image """
    return np.max(image)

def calculate_lsf(roi):
    """ Calculate Line Spread Function (LSF) using edge detection """
    edges = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=5)
    lsf = np.mean(edges, axis=0)
    return lsf

def calculate_esf(lsf):
    """ Calculate Edge Spread Function (ESF) from LSF """
    esf = np.cumsum(lsf)
    return esf

def calculate_mtf(esf):
    """ Calculate Modulation Transfer Function (MTF) from ESF """
    mtf = np.abs(fft(esf))
    mtf = mtf / np.max(mtf)  # Normalize MTF to range [0, 1]
    return mtf

def interpolate_data(x, y, num_points=100):
    """ Interpolate data using CubicSpline """
    cs = CubicSpline(x, y)
    x_interp = np.linspace(x.min(), x.max(), num=num_points)
    y_interp = cs(x_interp)
    return x_interp, y_interp

def mtf_interpolate_data(x, y, num_points=1000):
    """ Interpolate data using PchipInterpolator """
    f = PchipInterpolator(x, y)
    x_interp = np.linspace(x.min(), x.max(), num=num_points)
    y_interp = f(x_interp)
    return x_interp, y_interp

def renormalize_mtf(mtf_interp):
    """ Renormalize MTF after interpolation """
    max_value = np.max(mtf_interp)
    return mtf_interp / max_value

def find_outer_contour(image):
    """ Find outer contour of object in the image """
    try:
        # Check if the image is grayscale
        if len(image.shape) > 2:
            # Convert DICOM image to grayscale
            dicom_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            dicom_image_gray = image
        
        # Ensure the image is of type CV_8U
        dicom_image_gray = cv2.convertScaleAbs(dicom_image_gray)
        
        # Apply Canny edge detection
        edge_image = cv2.Canny(dicom_image_gray, 100, 200)
        
        # Find contours in the binary edge image
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find the outermost contour
        contour = max(contours, key=cv2.contourArea)

        return contour
  
    except Exception as e:
        print(f"An error occurred while detecting the outer edge: {e}")
        return None

def plot_lsf_esf_mtf_image(image, lsf, esf, mtf, ds, roi_x, roi_y, roi_size=16):
    """ Plot LSF, ESF, MTF curves and the DICOM image with contour """
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))

    # Plot DICOM image with contour and ROI square
    axs[1,0].imshow(image, cmap='gray')
    contour = find_outer_contour(image)
    axs[1,0].plot(contour[:, 0, 0], contour[:, 0, 1], '-r', linewidth=1)
    
    # Draw ROI square in red
    roi_rect = plt.Rectangle((roi_x - roi_size//2, roi_y - roi_size//2), roi_size, roi_size, 
                             linewidth=1, edgecolor='r', facecolor='none')
    axs[1,0].add_patch(roi_rect)
    
    axs[1,0].set_title('DICOM Image')
    axs[1,0].axis('off')

    # Interpolate LSF
    x_lsf = np.arange(len(lsf))
    x_lsf_interp, lsf_interp = interpolate_data(x_lsf, lsf)
    axs[0,1].plot(x_lsf_interp, lsf_interp)
    axs[0,1].set_title('Line Spread Function (LSF)')
    axs[0,1].set_xlabel('Pixel Position')
    axs[0,1].set_ylabel('Intensity')
    axs[0,1].grid()

    # Interpolate ESF
    x_esf = np.arange(len(esf))
    x_esf_interp, esf_interp = interpolate_data(x_esf, esf)
    axs[1,1].plot(x_esf_interp, esf_interp)
    axs[1,1].set_title('Edge Spread Function (ESF)')
    axs[1,1].set_xlabel('Pixel Position')
    axs[1,1].set_ylabel('Cumulative Intensity')
    axs[1,1].grid()

    # Interpolate MTF
    x_mtf = np.fft.fftfreq(len(mtf))
    x_mtf_interp, mtf_interp = mtf_interpolate_data(x_mtf[:len(mtf)//2], mtf[:len(mtf)//2])
    mtf_interp_normalized = renormalize_mtf(mtf_interp)
    axs[2,1].plot(x_mtf_interp, mtf_interp_normalized)
    axs[2,1].set_title('Modulation Transfer Function (MTF)')
    axs[2,1].set_xlabel('Spatial Frequency')
    axs[2,1].set_ylabel('MTF')
    axs[2,1].grid()

    # Set y-axis ticks for MTF plot
    axs[2,1].set_yticks(np.arange(0, 1.1, 0.1))

    # Hide unused axis
    axs[0,0].axis('off')
    axs[2,0].axis('off')

    plt.tight_layout()
    plt.show()

def calculate_spatial_frequencies(mtf_interp, x_interp):
    """ Calculate spatial frequencies corresponding to specific MTF values """
    spatial_freq_values = {}
    
    # Find spatial frequencies corresponding to MTF values of interest
    for mtf_value in [0.50, 0.25, 0.10, 0.02]:
        # Find the nearest index where MTF is closest to the desired value
        idx = np.argmin(np.abs(mtf_interp - mtf_value))
        spatial_freq_values[mtf_value] = x_interp[idx]
    
    return spatial_freq_values

def analysis_folder(folder_path, roi_size=16):
    """ Process all DICOM images in a folder, calculate MTF, and print spatial frequencies """
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {file_path}")
        image, ds = load_dicom_image(file_path)

        # Find maximum HU value
        max_hu_value = find_max_hu_value(image)

        # Find coordinates of the maximum HU value in the image
        high_hu_pixels = np.where(image == max_hu_value)
        roi_x, roi_y = high_hu_pixels[1][0], high_hu_pixels[0][0]

        # Extract ROI around the high HU pixel
        roi = image[roi_y - roi_size//2 : roi_y + roi_size//2,
                       roi_x - roi_size//2 : roi_x + roi_size//2]

        # Calculate LSF (Line Spread Function)
        lsf = calculate_lsf(roi)

        # Calculate ESF (Edge Spread Function)
        esf = calculate_esf(lsf)

        # Calculate MTF (Modulation Transfer Function)
        mtf = calculate_mtf(esf)

        # Interpolate MTF data
        x_mtf = np.fft.fftfreq(len(mtf))
        x_mtf_interp, mtf_interp = mtf_interpolate_data(x_mtf[:len(mtf)//2], mtf[:len(mtf)//2])

        # Print spatial frequencies corresponding to specific MTF values
        spatial_freq_values = calculate_spatial_frequencies(mtf_interp, x_mtf_interp)
        for mtf_value, spatial_freq in spatial_freq_values.items():
            print(f"Spatial frequency at MTF {mtf_value*100}%: {spatial_freq:.2f}")

        # Plot LSF, ESF, MTF curves and DICOM image
        plot_lsf_esf_mtf_image(image, lsf, esf, mtf, ds, roi_x, roi_y, roi_size)

# Input folder path and ROI size
folder_path = r'D:\Github\MPH3013-Special-Project\Coding\Images to be processed'
roi_size = int(input("Enter ROI size (integer value): "))

analysis_folder(folder_path, roi_size)
print('Analysis of all images is complete.')
