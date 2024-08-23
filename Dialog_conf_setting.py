from functools import partial
from PyQt5.QtWidgets import QGridLayout, QSlider, QPushButton, QLabel, QHBoxLayout, QVBoxLayout, QDialog
from PyQt5 import QtCore
class ConfidenceSettingsDialog(QDialog):
    def __init__(self, confidence_levels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Confidence Levels")
        self.confidence_levels = confidence_levels
        self.setLayout(QVBoxLayout())

        self.sliders = {}
        for label, conf in self.confidence_levels.items():
            horizontal_layout = QHBoxLayout()
            
            slider_label = QLabel(f"{label} Confidence: {conf:.2f}")
            horizontal_layout.addWidget(slider_label)
            
            slider = QSlider(QtCore.Qt.Horizontal, self)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(int(conf * 100))
            slider.valueChanged.connect(partial(self.update_confidence, label=label, slider_label=slider_label))
            horizontal_layout.addWidget(slider)
            
            self.sliders[label] = slider
            self.layout().addLayout(horizontal_layout)

        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK", self)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        self.layout().addLayout(button_layout)

    def update_confidence(self, value, label, slider_label):
        self.confidence_levels[label] = value / 100.0
        slider_label.setText(f"{label} Confidence: {self.confidence_levels[label]:.2f}")

    def get_confidence_levels(self):
        return self.confidence_levels