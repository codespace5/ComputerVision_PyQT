from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QAction, QFileDialog, QLabel, QWidget, QDialog, QLineEdit, QVBoxLayout, QPushButton, QFormLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import QCoreApplication, Qt, QEvent, QTimer
from database import Database
import sqlite3

# Sign Up Dialog
class SignUpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Sign Up')
        self.db = Database()
        self.setFixedSize(500, 500)
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)

        self.form_layout = QFormLayout()
        self.username_input = QLineEdit(self)
        self.farm_name_input = QLineEdit(self)
        self.address_input = QLineEdit(self)
        self.user_id_input = QLineEdit(self)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)

        self.form_layout.addRow("Username", self.username_input)
        self.form_layout.addRow("Farm Name", self.farm_name_input)
        self.form_layout.addRow("Address", self.address_input)
        self.form_layout.addRow("User ID", self.user_id_input)
        self.form_layout.addRow("Password", self.password_input)

        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.register_user)
        
        self.layout.addLayout(self.form_layout)
        self.layout.addWidget(self.ok_button)

    def register_user(self):
        username = self.username_input.text().strip()
        farm_name = self.farm_name_input.text().strip()
        address = self.address_input.text().strip()
        user_id = self.user_id_input.text().strip()
        password = self.password_input.text().strip()

        if not all([username, farm_name, address, user_id, password]):
            QMessageBox.warning(self, "Error", "All fields are required!")
            return

        if self.db.get_password(user_id):
            QMessageBox.warning(self, "Error", "User ID already exists!")
            return

        self.db.register_user(username, farm_name, address, user_id, password)
        QMessageBox.information(self, "Success", "User registered! Please login.")
        self.accept()

# Login Dialog
class LoginDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Login')
        self.db = Database()
        self.setFixedSize(400, 200)
        self.init_ui()

    def init_ui(self):
        self.layout = QFormLayout(self)
        self.user_id_input = QLineEdit(self)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)
        
        self.login_button = QPushButton("Login", self)
        self.sign_up_button = QPushButton("Sign Up", self)

        self.layout.addRow("User ID", self.user_id_input)
        self.layout.addRow("Password", self.password_input)
        self.layout.addRow(self.login_button, self.sign_up_button)
        
        self.login_button.clicked.connect(self.confirm_login)
        self.sign_up_button.clicked.connect(self.sign_up)
        
    @property    
    def logged_in_user_id(self):
        user_id = self.user_id_input.text()
        return user_id  # Assuming `user_id` is where the ID is stored after a successful login
    
    def confirm_login(self):
        user_id = self.user_id_input.text().strip()
        password = self.password_input.text().strip()

        if not user_id or not password:
            QMessageBox.warning(self, "Error", "Please enter both User ID and Password!")
            return

        db_password = self.db.get_password(user_id)

        if not db_password:
            QMessageBox.warning(self, "Error", "User ID does not exist!")
            return

        if password != db_password:
            QMessageBox.warning(self, "Error", "Incorrect Password!")
            return

        self.accept()

    def sign_up(self):
        self.sign_up_dialog = SignUpDialog(self)
        self.sign_up_dialog.exec_()