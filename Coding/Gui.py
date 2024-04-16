import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QLineEdit, QFileDialog
import analysis_module

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Image Analysis")
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)
        
        # Square size input
        self.square_size_label = QLabel("Enter the size of the square ROI in pixels:")
        self.layout.addWidget(self.square_size_label)
        
        self.square_size_input = QLineEdit()
        self.square_size_input.setPlaceholderText("Enter square size")
        self.layout.addWidget(self.square_size_input)
        
        # Select DICOM folder button
        self.select_folder_button = QPushButton("Select DICOM Folder")
        self.select_folder_button.clicked.connect(self.select_folder)
        self.layout.addWidget(self.select_folder_button)
        
        # Analysis results label
        self.analysis_results_label = QLabel()
        self.layout.addWidget(self.analysis_results_label)
        
        # Process button
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process)
        self.layout.addWidget(self.process_button)
    
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder_path:
            self.folder_path = folder_path
    
    def process(self):
        square_size_text = self.square_size_input.text()
        if square_size_text.isdigit():
            square_size = int(square_size_text)
            if hasattr(self, 'folder_path'):
                analysis_results = analysis_module.process_dicom_folder(self.folder_path, square_size)
                self.display_results(analysis_results)
            else:
                self.analysis_results_label.setText("Please select a DICOM folder.")
        else:
            self.analysis_results_label.setText("Please enter a valid square size.")
    
    def display_results(self, results):
        # Update the GUI with analysis results
        if results:
            analysis_text = "Analysis Results:\n"
            for result in results:
                analysis_text += f"- {result['image_name']}: {result['analysis_result']}\n"
            self.analysis_results_label.setText(analysis_text)
        else:
            self.analysis_results_label.setText("No analysis results found.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
