import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy import fftpack

def draw_square_roi_on_dicom(dicom_file_path, square_origin, square_size):
    # Load DICOM file
    dicom = pydicom.dcmread(dicom_file_path)
    
    # Extract image data
    image = dicom.pixel_array
    
    # Create a figure to display the DICOM image
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title('DICOM Image with Square ROI')
    
    # Extract square coordinates
    x0, y0 = square_origin
    
    # Define the square ROI coordinates
    x1 = x0 + square_size
    y1 = y0 + square_size
    
    # Plot the rectangle ROI on the image
    rect = patches.Rectangle((x0, y0), square_size, square_size, linewidth=1, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    
    # Show the image with the square ROI
    plt.show()
    
    return x0, y0, x1, y1

def compute_pixel_spacing(dicom):
    """
    Compute the pixel spacing from DICOM metadata.

    Parameters:
    dicom (pydicom.dataset.FileDataset): DICOM dataset.

    Returns:
    float: Pixel spacing in mm.
    """
    if 'PixelSpacing' in dicom:
        return float(dicom.PixelSpacing[0])
    elif 'ImagerPixelSpacing' in dicom:
        return float(dicom.ImagerPixelSpacing[0])
    else:
        raise ValueError("Pixel spacing not found in DICOM metadata")

def compute_psf_from_square_roi(image, square_origin, square_size):
    """
    Compute the Point Spread Function (PSF) from a square Region of Interest (ROI).

    Parameters:
    image (numpy.ndarray): Array containing the image data.
    square_origin (tuple): Coordinates (x, y) of the top-left corner of the square ROI.
    square_size (int): Size of the square ROI in pixels.

    Returns:
    numpy.ndarray: Array containing the Point Spread Function (PSF).
    """
    # Extract pixels inside the square ROI
    x0, y0 = square_origin
    roi = image[y0:y0+square_size, x0:x0+square_size]
    
    # Compute the PSF by averaging intensity values along rows and columns
    psf = np.mean(roi, axis=(0, 1))  # Assuming a 2D ROI
    
    # Normalize the PSF
    psf /= np.sum(psf)
    
    return psf

def compute_mtf_from_psf(psf, pixel_spacing):
    """
    Compute the Modulation Transfer Function (MTF) from the Point Spread Function (PSF).

    Parameters:
    psf (numpy.ndarray): Array containing the Point Spread Function (PSF).
    pixel_spacing (float): Pixel spacing in the image (in mm).

    Returns:
    numpy.ndarray: Array containing spatial frequencies.
    numpy.ndarray: Array containing MTF values.
    """
    # Compute the Discrete Fourier Transform (DFT) of the PSF
    dft_psf = np.fft.fft2(psf)
    
    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(dft_psf)
    
    # Compute the spatial frequencies
    num_rows, num_cols = psf.shape
    frequencies_x = np.fft.fftfreq(num_cols, d=pixel_spacing)
    frequencies_y = np.fft.fftfreq(num_rows, d=pixel_spacing)
    
    # Normalize the magnitude spectrum
    normalized_magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
    
    return frequencies_x, frequencies_y, normalized_magnitude_spectrum

def plot_mtf_from_psf(psf, frequencies_x, frequencies_y, title="Modulation Transfer Function (MTF)"):
    print("Shapes of input arrays:")
    print("psf:", psf.shape)
    print("frequencies_x:", frequencies_x.shape)
    print("frequencies_y:", frequencies_y.shape)
    
    plt.plot(frequencies_x, psf, label='X direction')
    plt.plot(frequencies_y, psf, label='Y direction')
    plt.xlabel('Spatial Frequency (cycles/mm)')
    plt.ylabel('MTF')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def extract_pixels_inside_square_roi(dicom_file_path, square_origin, square_size):
    # Load DICOM file
    dicom = pydicom.dcmread(dicom_file_path)
    
    # Extract image data
    image = dicom.pixel_array
    
    # Extract square coordinates
    x0, y0 = square_origin
    
    # Extract pixels inside the square ROI
    pixels_inside_square = image[y0:y0+square_size, x0:x0+square_size]
    
    return pixels_inside_square

# Example usage
dicom_file_path = 'D:\Github\MPH3013-Special-Project\Coding\mdh_images\PHYSICS_BODY_CATPHAN.CT.ABDOMEN_ABDOMENSEQ_(ADULT).0003.0004.2023.02.08.12.00.21.950084.6467264.IMA'

point_location_x = 255
point_location_y = 368

square_size = 5  # Size of the square ROI in pixels

if square_size % 2 == 1 :
    square_size +=1

square_origin = (int((point_location_x-(0.5*square_size))), int((point_location_y-(0.5*square_size))))  # Coordinates of the top-left corner of the square ROI

draw_square_roi_on_dicom(dicom_file_path, square_origin, square_size)

# Load DICOM file
dicom = pydicom.dcmread(dicom_file_path)

# Automatically calculate pixel spacing
pixel_spacing = compute_pixel_spacing(dicom)

# Extract pixels inside the square ROI
pixels_inside_square = extract_pixels_inside_square_roi(dicom_file_path, square_origin, square_size)

psf = compute_psf_from_square_roi(pixels_inside_square, square_origin, square_size)

frequencies_x, frequencies_y, _ = compute_mtf_from_psf(psf, pixel_spacing)

plot_mtf_from_psf(psf, frequencies_x, frequencies_y)