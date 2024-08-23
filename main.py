import sys
import os
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QAction, QFileDialog, QLabel, QWidget, QDialog
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import QCoreApplication, Qt, QEvent, QTimer
from qt_material import apply_stylesheet
from Dialog_login import LoginDialog
from Dialog_api_key import SettingsDialog
from Dialog_conf_setting import ConfidenceSettingsDialog
from Dialog_notification_delay import NotificationDelayDialog

class ImageVideoDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Farm Monitoring System v1.11")

        # Initialize the starting GIF first
        self.start_gif_label = QLabel(self)
        self.setCentralWidget(self.start_gif_label)
        self.start_gif_movie = QMovie("./icons/loading.gif")
        self.start_gif_label.setMovie(self.start_gif_movie)
        self.start_gif_movie.start()

        self.webcam_connected = False
        self.detect_script_path = ""

        # Create Menu Bar
        self.menu_bar = self.menuBar()
        
        
        # Create user ID label - initially empty
        self.user_id_label = QLabel("")
        self.user_id_label.setStyleSheet("color: Orange; font-size: 16pt; font-family: Comic Sans;")
        self.menu_bar.setCornerWidget(self.user_id_label, Qt.TopRightCorner)
        
        # Fire Detection Menu
        fire_menu = self.menu_bar.addMenu("Fire Detection")
        self.create_detection_menu(fire_menu, "Fire")

        # People Detection Menu
        people_menu = self.menu_bar.addMenu("People Detection")
        self.create_detection_menu(people_menu, "People")

        # Cattle Detection Menu
        cattle_menu = self.menu_bar.addMenu("Cattle Detection")
        self.create_detection_menu(cattle_menu, "Cattle")

        # Sheep Detection Menu
        sheep_menu = self.menu_bar.addMenu("Sheep Detection")
        self.create_detection_menu(sheep_menu, "Sheep")
        
        #Confidence Score Setting Option
        self.confidence_levels = {"Fire": 0.45, "People": 0.25, "Cattle": 0.25, "Sheep": 0.25}
        settings_menu = self.menu_bar.addMenu("Settings")
        self.confidence_action = QAction("Set Confidence Levels", self)
        self.confidence_action.triggered.connect(self.open_confidence_settings)
        settings_menu.addAction(self.confidence_action)
        self.notification_delay = 5  # Default value in seconds
        self.notification_delay_action = QAction("Set Mobile Notification Delay", self)
        self.notification_delay_action.triggered.connect(self.open_notification_delay_settings)
        settings_menu.addAction(self.notification_delay_action)
        # Hide the menu bar initially
        self.menu_bar.setVisible(False)

        QTimer.singleShot(3000, self.show_main_window)

    def show_main_window(self):
        login_dialog = LoginDialog()
        result = login_dialog.exec_()

        if result == QDialog.Accepted:
            # get the user's ID and display it on the menu bar
            user_id = login_dialog.logged_in_user_id
            self.user_id_label.setText(f"User ID: {user_id}")
            self.user_id_action = QAction(self.user_id_label.text(), self)
            self.user_id_action.triggered.connect(self.open_settings)
            self.menu_bar.addAction(self.user_id_action)
            
            self.start_gif_movie.stop()
            background_image = QPixmap('./icons/background.jpg')
            self.background_label = QLabel(self)
            self.background_label.setPixmap(background_image)
            self.background_label.setGeometry(0, 0, background_image.width(), background_image.height())
            self.setCentralWidget(self.background_label)
            self.menu_bar.setVisible(True)
        else:
            QCoreApplication.quit()
            
    def open_confidence_settings(self):
        dialog = ConfidenceSettingsDialog(self.confidence_levels, self)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            self.confidence_levels = dialog.get_confidence_levels()
            
    def open_settings(self):
        settings_dialog = SettingsDialog(self.user_id_label.text())
        settings_dialog.exec_()
        
    def open_notification_delay_settings(self):
        dialog = NotificationDelayDialog(self.notification_delay, self)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            self.notification_delay = dialog.get_delay() if dialog.get_delay() else self.notification_delay
        
    def create_detection_menu(self, parent_menu, label):
        load_action = QAction(f"Load {label} Video", self)
        load_action.triggered.connect(lambda: self.load_file(label))
        parent_menu.addAction(load_action)

        connect_action = QAction(f"Connect {label} Webcam", self)
        connect_action.triggered.connect(lambda: self.connect_webcam(label))
        parent_menu.addAction(connect_action)

    def load_file(self, label):
        file_path, _ = QFileDialog.getOpenFileName(self, f"Open {label} Video File")
        if file_path:
            self.execute_detection(file_path, label)

    def connect_webcam(self, label):
        if not self.webcam_connected:
            command = self.get_webcam_command(label)
            self.process = subprocess.Popen(command, shell=True)
            self.webcam_connected = True
        else:
            self.process.terminate()
            self.webcam_connected = False

    def execute_detection(self, file_path, label):
        abs_file_path = os.path.abspath(file_path)
        command = self.get_detection_command(abs_file_path, label)
        subprocess.run(command, shell=True)

    def get_detection_command(self, file_path, label):
        conf = self.confidence_levels.get(label)
        delay_arg = f"--delay {self.notification_delay}"
        if label == "Fire":
            return f"python ./detect.py --source \"{file_path}\" --img 640 --weights ./fire_best.pt --conf {conf} --view-img {delay_arg}"
        elif label == "People":
            return f"python ./detect.py --weights yolov5s.pt --img 640 --conf {conf} --source \"{file_path}\" --view-img --classes 0 {delay_arg}"
        elif label == "Cattle":
            return f"python ./detect.py --weights yolov5s.pt --img 640 --conf {conf} --source \"{file_path}\" --view-img --classes 19 {delay_arg}"
        elif label == "Sheep":
            return f"python ./detect.py --weights yolov5s.pt --img 640 --conf {conf} --source \"{file_path}\" --view-img --classes 18 {delay_arg}"

    def get_webcam_command(self, label):
        conf = self.confidence_levels.get(label)
        delay_arg = f"--delay {self.notification_delay}"
        if label == "Fire":
            return f"python ./detect.py --source 0 --img 640 --weights ./fire_best.pt --device cpu --conf {conf} --view-img {delay_arg}"
        elif label == "People":
            return f"python ./detect.py --source 0 --weights yolov5s.pt --img 640 --conf {conf}  --view-img --classes 0 {delay_arg}"
        elif label == "Cattle":
            return f"python ./detect.py --source 0 --weights yolov5s.pt --img 640 --conf {conf}  --view-img --classes 19 {delay_arg}"
        elif label == "Sheep":
            return f"python ./detect.py --source 0 --weights yolov5s.pt --img 640 --conf {conf}  --view-img --classes 18 {delay_arg}"

    def closeEvent(self, event):
        self.menu_bar.setVisible(False)
        if self.webcam_connected:
            self.process.terminate()

        # Play the ending GIF once again and close the application after 5 seconds
        self.ending_gif_label = QLabel(self)
        self.setCentralWidget(self.ending_gif_label)
        self.ending_gif_movie = QMovie("./icons/ending.gif")
        self.ending_gif_label.setMovie(self.ending_gif_movie)
        self.ending_gif_movie.start()
        
        QTimer.singleShot(5000, QCoreApplication.quit)
        event.ignore()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Q and event.modifiers() == Qt.ControlModifier:
                QCoreApplication.quit()
        return super().eventFilter(obj, event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageVideoDetectorApp()
    window.setGeometry(200, 200, 800, 600)
    apply_stylesheet(app, theme='dark_teal.xml')
    window.show()
    app.installEventFilter(window)
    sys.exit(app.exec_())