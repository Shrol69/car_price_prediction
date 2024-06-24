import sys
import pandas as pd
import joblib
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox, QHBoxLayout, QFormLayout
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class CarPricePredictionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Car Price Prediction')
        self.setGeometry(200, 200, 800, 400)  # Adjusted window size for better visibility
        self.setStyleSheet("background-color: #f0f0f0;")
        self.initUI()
        
        # Load the trained model
        self.model = joblib.load('car_price_prediction_model.pkl')
        
    def initUI(self):
        main_layout = QHBoxLayout()
        
        # Left side layout for image and form
        left_layout = QVBoxLayout()
        
        # Load and display image
        image_label = QLabel()
        pixmap = QPixmap('News-Used-car-prices-could-decline.webp')
        pixmap = pixmap.scaledToWidth(300)  # Adjust width as needed
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(image_label)
        
        # Form layout for inputs
        form_layout = QFormLayout()
        
        # Car Name
        self.car_name_input = QLineEdit()
        form_layout.addRow('Car Name:', self.car_name_input)
        
        # Year
        self.year_input = QLineEdit()
        form_layout.addRow('Year:', self.year_input)
        
        # Present Price
        self.present_price_input = QLineEdit()
        form_layout.addRow('Present Price (in lakhs):', self.present_price_input)
        
        # Driven Kilometers
        self.driven_kms_input = QLineEdit()
        form_layout.addRow('Driven Kilometers:', self.driven_kms_input)
        
        # Fuel Type
        self.fuel_type_input = QLineEdit()
        form_layout.addRow('Fuel Type:', self.fuel_type_input)
        
        # Selling Type
        self.selling_type_input = QLineEdit()
        form_layout.addRow('Selling Type:', self.selling_type_input)
        
        # Transmission
        self.transmission_input = QLineEdit()
        form_layout.addRow('Transmission:', self.transmission_input)
        
        # Owner
        self.owner_input = QLineEdit()
        form_layout.addRow('Owner:', self.owner_input)
        
        # Predict Button
        self.predict_button = QPushButton('Predict Selling Price')
        self.predict_button.clicked.connect(self.predict_price)
        self.predict_button.setStyleSheet("""
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 10px;
            border: none;
            border-radius: 5px;
        """)
        
        # Add form layout to left side layout
        left_layout.addLayout(form_layout)
        left_layout.addWidget(self.predict_button, alignment=Qt.AlignCenter)
        
        # Add left side layout to main layout
        main_layout.addLayout(left_layout)
        
        self.setLayout(main_layout)
        
    def predict_price(self):
        try:
            car_data = {
                'Car_Name': [self.car_name_input.text()],
                'Year': [int(self.year_input.text())],
                'Present_Price': [float(self.present_price_input.text())],
                'Driven_kms': [int(self.driven_kms_input.text())],
                'Fuel_Type': [self.fuel_type_input.text()],
                'Selling_type': [self.selling_type_input.text()],
                'Transmission': [self.transmission_input.text()],
                'Owner': [int(self.owner_input.text())]
            }
            
            car_df = pd.DataFrame(car_data)
            
            # Use the loaded model to predict
            predicted_price = self.model.predict(car_df)[0]
            
            # Display the predicted price
            msg = QMessageBox()
            msg.setWindowTitle('Prediction Result')
            msg.setText(f'Predicted Selling Price: {predicted_price:.2f} lakhs')
            msg.exec_()
            
        except Exception as e:
            msg = QMessageBox()
            msg.setWindowTitle('Error')
            msg.setText(f'Error: {str(e)}')
            msg.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CarPricePredictionApp()
    window.show()
    sys.exit(app.exec_())
