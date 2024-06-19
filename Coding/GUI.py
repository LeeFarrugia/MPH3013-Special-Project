#gui

import tkinter as tk
from tkinter import filedialog, messagebox
from analysis_module import analysis_folder

def select_folder():
    folder_path = filedialog.askdirectory()
    folder_path_entry.delete(0, tk.END)
    folder_path_entry.insert(0, folder_path)

def select_output_folder():
    output_folder = filedialog.askdirectory()
    output_folder_entry.delete(0, tk.END)
    output_folder_entry.insert(0, output_folder)

def run_analysis():
    folder_path = folder_path_entry.get()
    roi_size = roi_size_entry.get()
    output_folder = output_folder_entry.get()
    
    if not folder_path:
        messagebox.showerror("Error", "Please select a folder containing DICOM images.")
        return
    
    if not output_folder:
        messagebox.showerror("Error", "Please specify an output folder.")
        return

    try:
        roi_size = int(roi_size)
    except ValueError:
        messagebox.showerror("Error", "ROI size must be an integer.")
        return

    try:
        analysis_folder(folder_path, roi_size, output_folder=output_folder)
        messagebox.showinfo("Success", "MTF analysis completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during analysis: {e}")

# Create the main window
root = tk.Tk()
root.title("MTF Analysis Tool")

# Folder selection
tk.Label(root, text="Folder Path:").grid(row=0, column=0, padx=10, pady=10)
folder_path_entry = tk.Entry(root, width=50)
folder_path_entry.grid(row=0, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=select_folder).grid(row=0, column=2, padx=10, pady=10)

# ROI size input
tk.Label(root, text="ROI Size:").grid(row=1, column=0, padx=10, pady=10)
roi_size_entry = tk.Entry(root, width=10)
roi_size_entry.grid(row=1, column=1, padx=10, pady=10)
roi_size_entry.insert(0, "16")

# Output folder selection
tk.Label(root, text="Output Folder:").grid(row=2, column=0, padx=10, pady=10)
output_folder_entry = tk.Entry(root, width=50)
output_folder_entry.grid(row=2, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=select_output_folder).grid(row=2, column=2, padx=10, pady=10)

# Run analysis button
tk.Button(root, text="Run Analysis", command=run_analysis).grid(row=3, column=0, columnspan=3, padx=10, pady=20)

# Start the main event loop
root.mainloop()
