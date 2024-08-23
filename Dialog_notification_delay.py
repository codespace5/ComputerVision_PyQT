from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QHBoxLayout

class NotificationDelayDialog(QDialog):
    def __init__(self, delay, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Notification Delay")
        self.setLayout(QVBoxLayout())

        self.info_label = QLabel(f"Set delay (in seconds) for mobile notifications:", self)
        self.layout().addWidget(self.info_label)

        self.delay_input = QLineEdit(self)
        self.delay_input.setText(str(delay))
        self.layout().addWidget(self.delay_input)

        self.buttons_layout = QHBoxLayout()
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        self.buttons_layout.addWidget(self.save_button)
        self.buttons_layout.addWidget(self.cancel_button)
        self.layout().addLayout(self.buttons_layout)

    def get_delay(self):
        try:
            return int(self.delay_input.text().strip())
        except ValueError:
            return None