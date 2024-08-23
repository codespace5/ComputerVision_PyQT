from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QVBoxLayout, QPushButton, QMessageBox
from database import Database

class SettingsDialog(QDialog):
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id
        self.setWindowTitle("User Settings")
        self.setGeometry(300, 300, 300, 150)
        
        # Setting up the layout
        layout = QVBoxLayout()
        
        self.api_key_label = QLabel("Enter API Key for Mobile App Notification:", self)
        self.api_key_input = QLineEdit(self)
        
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.save_api_key)
        
        layout.addWidget(self.api_key_label)
        layout.addWidget(self.api_key_input)
        layout.addWidget(self.ok_button)
        
        self.setLayout(layout)

    def save_api_key(self):
        db = Database()
        api_key = self.api_key_input.text().strip()

        # Check if the api key is already registered for this user
        if db.is_api_key_registered(self.user_id, api_key):
            QMessageBox.warning(self, "Warning", "The API Key is already registered for this user.")
            return
        
        # Update the user's api key in the database
        db.register_api_key(self.user_id, api_key)
        QMessageBox.information(self, "Success", "API Key successfully registered!")
        self.accept()