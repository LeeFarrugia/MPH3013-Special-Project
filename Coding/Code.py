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

def compute_mtf_from_lsf(lsf, pixel_spacing):
    """
    Compute the Modulation Transfer Function (MTF) from the Line Spread Function (LSF).

    Parameters:
    lsf (numpy.ndarray): Array containing intensity values along the line (LSF).
    pixel_spacing (float): Pixel spacing in the image (in mm).

    Returns:
    numpy.ndarray: Array containing spatial frequencies.
    numpy.ndarray: Array containing MTF values.
    """
    # Calculate spatial frequencies
    num_pixels = len(lsf)
    frequencies = fftpack.fftfreq(num_pixels, d=pixel_spacing)
    
    # Perform Fourier Transform of the LSF
    lsf_fft = fftpack.fft(lsf)
    
    # Calculate MTF (absolute value of the Fourier Transform)
    mtf = np.abs(lsf_fft)
    
    return frequencies, mtf

def compute_lsf_from_pixels(pixels):
    """
    Compute the Line Spread Function (LSF) from the pixels inside the square ROI.

    Parameters:
    pixels (numpy.ndarray): Array containing pixel values inside the square ROI.

    Returns:
    numpy.ndarray: Array containing intensity values along the line.
    """
    # Calculate the center of the square ROI
    center_x = pixels.shape[1] // 2
    center_y = pixels.shape[0] // 2
    
    # Compute the LSF along the vertical and horizontal axes
    lsf_horizontal = pixels[center_y, :]
    lsf_vertical = pixels[:, center_x]
    
    return lsf_horizontal, lsf_vertical

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

# Compute the Line Spread Function (LSF) from the pixels inside the square ROI
lsf_horizontal, lsf_vertical = compute_lsf_from_pixels(pixels_inside_square)

# Compute the Modulation Transfer Function (MTF) from the Line Spread Function (LSF)
frequencies, mtf_horizontal = compute_mtf_from_lsf(lsf_horizontal, pixel_spacing)
frequencies, mtf_vertical = compute_mtf_from_lsf(lsf_vertical, pixel_spacing)

# Plot the LSF
plt.plot(lsf_horizontal, label='Horizontal LSF')
plt.plot(lsf_vertical, label='Vertical LSF')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')
plt.title('Line Spread Function (LSF) from Pixels Inside Square ROI')
plt.legend()
plt.grid(True)
plt.show()

# Plot the MTF
plt.plot(frequencies, mtf_horizontal, label='Horizontal MTF')
plt.xlabel('Spatial Frequency (cycles/mm)')
plt.ylabel('MTF')
plt.title('Modulation Transfer Function (MTF) from Line Spread Function (LSF)')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(frequencies, mtf_vertical, label='Vertical MTF')
plt.xlabel('Spatial Frequency (cycles/mm)')
plt.ylabel('MTF')
plt.title('Modulation Transfer Function (MTF) from Line Spread Function (LSF)')
plt.legend()
plt.grid(True)
plt.show()
