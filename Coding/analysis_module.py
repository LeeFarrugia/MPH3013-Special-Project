# analysis_module.py

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fft import fft
import cv2
from scipy.interpolate import PchipInterpolator, CubicSpline
from collections import defaultdict

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

def mtf_interpolate_data(x, y, num_points=1000, scale_factor=10):
    """ Interpolate data using PchipInterpolator and convert to cycles/cm """
    f = PchipInterpolator(x * scale_factor, y)  # Convert x-axis to cycles/cm
    x_interp = np.linspace((x * scale_factor).min(), (x * scale_factor).max(), num=num_points)
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

def plot_lsf_esf_mtf_image(image, lsf, esf, mtf, ds, roi_x, roi_y, std_sr, roi_size=16):
    """ Plot LSF, ESF, MTF curves and the DICOM image with contour """
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))

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
    axs[0,1].plot(x_lsf_interp, lsf_interp, color ='k')
    axs[0,1].set_title('Line Spread Function (LSF)')
    axs[0,1].set_xlabel('Pixel Position')
    axs[0,1].set_ylabel('Intensity (HU)')
    axs[0,1].grid()

    # Interpolate ESF
    x_esf = np.arange(len(esf))
    x_esf_interp, esf_interp = interpolate_data(x_esf, esf)
    axs[1,1].plot(x_esf_interp, esf_interp, color ='k')
    axs[1,1].set_title('Edge Spread Function (ESF)')
    axs[1,1].set_xlabel('Pixel Position')
    axs[1,1].set_ylabel('Cumulative Intensity (HU)')
    axs[1,1].grid()

    # Interpolate MTF
    x_mtf = np.fft.fftfreq(len(mtf))
    x_mtf_interp, mtf_interp = mtf_interpolate_data(x_mtf[:len(mtf)//2], mtf[:len(mtf)//2])
    mtf_interp_normalized = renormalize_mtf(mtf_interp)
    axs[2,1].plot(x_mtf_interp, mtf_interp_normalized, color ='k')
    axs[2,1].set_title('Modulation Transfer Function (MTF)')
    axs[2,1].set_xlabel('Spatial Frequency (cycles/cm)')
    axs[2,1].set_ylabel('MTF')
    axs[2,1].grid()

    # Set y-axis ticks for MTF plot
    axs[2,1].set_yticks(np.arange(0, 1.1, 0.1))

    # Hide unused axis
    axs[0,0].axis('off')
    axs[2,0].axis('off')

    # Calculate spatial frequencies corresponding to specific MTF values
    spatial_freq_values = calculate_spatial_frequencies(mtf_interp, x_mtf_interp)
    
    # Add spatial frequencies as a table
    table_data = [["MTF 50%", f"{spatial_freq_values[0.50]:.2f}  \u00B1 {std_sr:.2f} cycles/cm"],
                  ["MTF 10%", f"{spatial_freq_values[0.10]:.2f}  \u00B1 {std_sr:.2f} cycles/cm"],
                  ["MTF 2%", f"{spatial_freq_values[0.02]:.2f}  \u00B1 {std_sr:.2f} cycles/cm"]]
    
    table = axs[2,0].table(cellText=table_data,
                          colLabels=["MTF (%)", "Spatial Frequency"],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.2, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.tight_layout()

    return fig

def plot_all_mtf_data(all_mtf_data, output_folder, filename, scale_factor=10):
    """ Plot all MTF data from every image into one plot in grey, and the average of all data points in black with selected std deviation error bars """
    plt.figure(figsize=(10, 6))

    # Plot all MTF curves in grey
    for x_mtf_interp, mtf_interp in all_mtf_data:
        plt.plot(x_mtf_interp * scale_factor, mtf_interp, color='grey', alpha=0.5)

    # Interpolate to common x-values
    x_values = np.linspace(0, np.max([x.max() for x, _ in all_mtf_data]) * scale_factor, 1000)
    interpolated_mtf_data = []

    for x_mtf_interp, mtf_interp in all_mtf_data:
        f = PchipInterpolator(x_mtf_interp * scale_factor, mtf_interp)
        interpolated_mtf_data.append(f(x_values))

    interpolated_mtf_data = np.array(interpolated_mtf_data)

    # Calculate the average and standard deviation of the interpolated MTF data
    avg_mtf = np.mean(interpolated_mtf_data, axis=0)
    std_sr = np.std(interpolated_mtf_data, axis=0)

    # Plot the average MTF curve in black
    plt.plot(x_values, avg_mtf, color='black', linewidth=2, label='Average MTF')

    # Define specific MTF positions
    mtf_positions = [0.50, 0.10, 0.02]

    # Find the indices closest to these MTF positions
    indices = [np.argmin(np.abs(avg_mtf - pos)) for pos in mtf_positions]

    # Plot error bars at these positions
    for idx in indices:
        plt.errorbar(x_values[idx], avg_mtf[idx], yerr=std_sr[idx], fmt='o', color='grey', capsize=5)

    plt.title('Combined MTF Plot')
    plt.xlabel('Spatial Frequency (cycles/cm)')
    plt.ylabel('MTF')
    plt.legend()
    plt.grid()

    # Save the combined MTF plot to a PDF file
    output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}combined_mtf_plot.pdf")
    with PdfPages(output_file) as pdf:
        pdf.savefig()
        plt.close()
    
    return std_sr

def calculate_spatial_frequencies(mtf_interp, x_interp):
    """ Calculate spatial frequencies corresponding to specific MTF values """
    spatial_freq_values = {}
    
    # Find spatial frequencies corresponding to MTF values of interest
    for mtf_value in [0.50, 0.10, 0.02]:
        # Find the nearest index where MTF is closest to the desired value
        idx = np.argmin(np.abs(mtf_interp - mtf_value))
        spatial_freq_values[mtf_value] = x_interp[idx]
    
    return spatial_freq_values

def analysis_folder(folder_path, roi_size=16, output_folder=None):
    """ Process all DICOM images in a folder, calculate MTF """
    if output_folder is None:
        output_folder = os.getcwd()
    os.makedirs(output_folder, exist_ok=True)

    all_mtf_data = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {file_path}")
        image, ds = load_dicom_image(file_path)

        max_hu_value = find_max_hu_value(image)
        high_hu_pixels = np.where(image == max_hu_value)
        roi_x, roi_y = high_hu_pixels[1][0], high_hu_pixels[0][0]

        roi = image[roi_y - roi_size//2 : roi_y + roi_size//2,
                       roi_x - roi_size//2 : roi_x + roi_size//2]

        lsf = calculate_lsf(roi)
        esf = calculate_esf(lsf)
        mtf = calculate_mtf(esf)

        x_mtf = np.fft.fftfreq(len(mtf))
        x_mtf_interp, mtf_interp = mtf_interpolate_data(x_mtf[:len(mtf)//2], mtf[:len(mtf)//2], scale_factor=10)

        all_mtf_data.append((x_mtf_interp, mtf_interp))
        std_sr = np.std(x_mtf_interp, axis=0)

        fig = plot_lsf_esf_mtf_image(image, lsf, esf, mtf, ds, roi_x, roi_y, std_sr, roi_size)

        output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_analysis.pdf")
        with PdfPages(output_file) as pdf:
            pdf.savefig(fig)
            plt.close(fig)
    
    # After processing all images, plot all MTF data together and get std_mtf
    plot_all_mtf_data(all_mtf_data, output_folder, filename, scale_factor=10)
