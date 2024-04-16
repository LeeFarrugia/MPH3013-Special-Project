# my_gui.py

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog
from analysis_module import process_dicom_folder

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Image Analysis")
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)
        
        self.select_folder_button = QPushButton("Select DICOM Folder")
        self.select_folder_button.clicked.connect(self.select_folder)
        self.layout.addWidget(self.select_folder_button)
        
        self.analysis_results_label = QLabel()
        self.layout.addWidget(self.analysis_results_label)
    
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder_path:
            square_size = 100  # Adjust as needed
            analysis_results = process_dicom_folder(folder_path, square_size)
            self.display_results(analysis_results)
    
    def display_results(self, results):
        # Update the GUI with analysis results
        self.analysis_results_label.setText("Analysis results here")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
