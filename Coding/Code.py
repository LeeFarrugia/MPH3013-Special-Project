import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydicom.errors import InvalidDicomError
from scipy.optimize import curve_fit

def calculate_ct_numbers_roi(dicom_file_path, square_origin, square_size):
    # Load DICOM file
    dicom_data = pydicom.dcmread(dicom_file_path)
    
    # Extract image data
    image = dicom_data.pixel_array
    
    # Extract Rescale Intercept and Rescale Slope
    rescale_intercept = dicom_data.RescaleIntercept
    rescale_slope = dicom_data.RescaleSlope
    
    # Extract pixel spacing
    pixel_spacing = dicom_data.PixelSpacing
    if pixel_spacing is None or len(pixel_spacing) != 2:
        raise ValueError("Pixel spacing information is missing or incomplete.")
    pixel_spacing_x, pixel_spacing_y = map(float, pixel_spacing)
    
    # Calculate the sampling distance (cm per sample)
    sampling_distance = max(pixel_spacing_x, pixel_spacing_y)
    
    # Extract square coordinates
    x0, y0 = square_origin
    
    # Define the square ROI coordinates
    x1 = x0 + square_size
    y1 = y0 + square_size
    
    # Crop the image to the ROI
    roi_image = image[y0:y1, x0:x1]
    
    # Calculate CT numbers for each pixel in the ROI
    ct_numbers_roi = (roi_image * rescale_slope) + rescale_intercept
    
    return ct_numbers_roi, sampling_distance

def generate_psf_from_roi(ct_numbers_roi):
    """
    Generate the Point Spread Function (PSF) from the ROI CT numbers.
    """
    # Calculate the Line Spread Function (LSF)
    lsf = np.diff(ct_numbers_roi, axis=1)
    
    # Integrate to get the Point Spread Function (PSF)
    psf = np.cumsum(lsf, axis=1)
    
    # Flatten the PSF to 1D
    psf = psf.flatten()
    
    return psf

def gaussian(x, amplitude, mean, stddev):
    """
    Gaussian function.
    """
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

def fit_gaussian_to_psf(psf, sampling_distance):
    """
    Fit a Gaussian curve to the Point Spread Function (PSF) and extract FWHM.
    """
    # Compute the mean value
    mean_value = np.sum(psf * np.arange(len(psf))) / np.sum(psf)
    
    # Initial guess for parameters
    amplitude_guess = np.max(psf)
    stddev_guess = len(psf) / 10  # Initial guess based on the size of the PSF
    
    # Fit the Gaussian curve
    popt, _ = curve_fit(gaussian, np.arange(len(psf)), psf, p0=[amplitude_guess, mean_value, stddev_guess])
    
    # Extract FWHM from the fitted Gaussian curve
    fwhm = 2 * np.sqrt(2 * np.log(2)) * popt[2] * sampling_distance
    
    return fwhm

def generate_lsf_from_fwhm(fwhm, n_points, sampling_distance):
    """
    Generate a Line Spread Function (LSF) from the Full Width at Half Maximum (FWHM).
    """
    # Calculate sigma from FWHM (FWHM = 2 * sqrt(2 * ln(2)) * sigma)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Generate x-axis values (equivalent to spatial coordinates)
    x_values = np.linspace(-3 * sigma, 3 * sigma, n_points)
    
    # Calculate LSF (Gaussian distribution)
    lsf = np.exp(-x_values**2 / (2 * sigma**2))
    
    # Normalize LSF
    lsf /= np.sum(lsf)
    
    return lsf

def calculate_mtf_from_lsf(lsf):
    """
    Calculate the Modulation Transfer Function (MTF) from the Line Spread Function (LSF).
    """
    mtf = np.abs(np.fft.fft(lsf))
    return mtf

def plot_lsf(ax, lsf, pixel_positions, sampling_distance, plot_type='line'):
    """
    Plot the Line Spread Function (LSF).
    """
    if plot_type == 'line':
        # Plot as a line graph
        ax.plot(pixel_positions, lsf)
        ax.set_xlabel('Pixel Position')
        ax.set_ylabel('LSF')
        ax.set_title('Line Spread Function (LSF) - Line Graph')
        ax.grid(True)
    elif plot_type == 'bar':
        # Plot as a bar graph
        ax.bar(pixel_positions, lsf, width=1)
        ax.set_xlabel('Pixel Position')
        ax.set_ylabel('LSF')
        ax.set_title('Line Spread Function (LSF) - Bar Graph')
    else:
        raise ValueError("Invalid plot type. Choose 'line' or 'bar'.")

def draw_square_roi_on_dicom(ax, dicom_file_path, square_origin, square_size):
    # Load DICOM file
    dicom_data = pydicom.dcmread(dicom_file_path)
    
    # Extract image data
    image = dicom_data.pixel_array
    
    # Plot the DICOM image
    ax.imshow(image, cmap='gray')
    ax.set_title('DICOM Image with Square ROI')
    
    # Extract square coordinates
    x0, y0 = square_origin
    
    # Define the square ROI coordinates
    x1 = x0 + square_size
    y1 = y0 + square_size
    
    # Plot the rectangle ROI on the image
    rect = patches.Rectangle((x0, y0), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

def plot_mtf(ax, lsf, sampling_distance):
    """
    Plot the Modulation Transfer Function (MTF).
    """
    # Calculate MTF from LSF
    mtf = calculate_mtf_from_lsf(lsf)
    
    # Plot the MTF
    frequencies = np.fft.fftfreq(len(mtf), d=sampling_distance)
    ax.plot(frequencies[:len(mtf) // 2], mtf[:len(mtf) // 2], label='MTF', color='red')
    ax.set_ylabel('MTF')
    ax.legend(loc='upper right')
    ax.grid(True)

def is_dicom_file(file_path):
    try:
        pydicom.dcmread(file_path)
        return True
    except (InvalidDicomError, IOError):
        return False

def process_dicom_folder(folder_path, square_size):
    # List image files in the folder
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if is_dicom_file(os.path.join(folder_path, file))]
    
    # Process each DICOM image file
    for dicom_file_path in image_files:
        # Coordinates of the top-left corner of the square ROI
        point_location_x = 255
        point_location_y = 368
        square_origin = (int((point_location_x - (0.5 * square_size))), int((point_location_y - (0.5 * square_size))))
        
        # Create a new figure for each DICOM image file
        fig, axs = plt.subplots(1, 3, figsize=(12, 6))
        
        # Draw square ROI on DICOM image
        draw_square_roi_on_dicom(axs[0], dicom_file_path, square_origin, square_size)
        
        # Automatically calculate FWHM from the ROI
        ct_numbers_roi, sampling_distance = calculate_ct_numbers_roi(dicom_file_path, square_origin, square_size - 1)
        psf = generate_psf_from_roi(ct_numbers_roi)
        fwhm_psf = fit_gaussian_to_psf(psf, sampling_distance)
        
        # Generate LSF from FWHM
        n_points = len(psf)
        lsf = generate_lsf_from_fwhm(fwhm_psf, n_points, sampling_distance)
        
        # Plot the LSF
        plot_lsf(axs[1], lsf, np.arange(len(lsf)), sampling_distance, plot_type='line')
        
        # Plot the MTF
        plot_mtf(axs[2], lsf, sampling_distance)
        
        # Title for the figure
        fig.suptitle('DICOM Image with ROI, LSF, MTF')
        
        # Show the figure
        plt.show()

# Input square size
square_size = int(input("Enter the size of the square ROI in pixels: "))

# Folder path containing DICOM files
folder_path = r'D:\Github\MPH3013-Special-Project\Coding\Images to be processed'

# Process the DICOM folder
process_dicom_folder(folder_path, square_size)
