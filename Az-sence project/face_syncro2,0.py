import sys
import cv2
import dlib
import numpy as np
import sqlite3
import os
import atexit
import platform
import subprocess
import datetime
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLineEdit, 
                            QLabel, QMessageBox, QListWidget, QListWidgetItem, 
                            QVBoxLayout, QWidget, QHBoxLayout, QInputDialog, 
                            QComboBox, QGroupBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# Color scheme
DARK_BLUE = "#2A2100"      # Dark golden background
PRIMARY_BLUE = "#D4A017"   # Primary golden color
ACCENT_RED = "#B22222"     # Accent red
WHITE = "#F5F5DC"          # Beige white
SECONDARY_TEXT = "#8B8000" # Secondary text color
BORDER_COLOR = "#DAA520"   # Golden border color

# Initialize Dlib models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def create_connection():
    """Create database connection"""
    conn = sqlite3.connect("users_dlib.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            name TEXT,
            descriptor BLOB,
            image BLOB,
            created_at TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            username TEXT PRIMARY KEY,
            password TEXT,
            role TEXT
        )
    """)
    return conn

class MainWindow(QMainWindow):
    def __init__(self, current_user):
        super().__init__()
        self.setWindowTitle("Face Recognition Attendance System")
        self.setGeometry(100, 100, 1200, 700)
        self.current_user = current_user
        self.conn = create_connection()
        self.cursor = self.conn.cursor()
        self.selected_user = None
        self.current_face_image = None
        self.last_recognized_name = None

        # Setup UI
        self.setup_ui()

        # Camera setup
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.current_face_descriptor = None

    def setup_ui(self):
        """Initialize user interface"""
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout(self.main_widget)
        
        # Left panel (controls)
        self.setup_left_panel()
        
        # Right panel (camera and status)
        self.setup_right_panel()

        # Add panels to main layout
        self.main_layout.addWidget(self.left_panel, 40)
        self.main_layout.addWidget(self.right_panel, 60)
        
        self.setCentralWidget(self.main_widget)

    def setup_left_panel(self):
        """Initialize left panel"""
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(10, 10, 10, 10)
        self.left_layout.setSpacing(15)
        
        # Control group
        self.setup_control_group()
        
        # Faces list
        self.setup_faces_list()
        
        self.left_layout.addStretch()

    def setup_control_group(self):
        """Initialize control group"""
        self.control_group = QGroupBox("Controls")
        self.control_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {PRIMARY_BLUE};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: {PRIMARY_BLUE};
                font-weight: bold;
                font-family: 'Poetsen One';
                font-size: 14pt;
                
            }}
           
        """)
        self.control_layout = QVBoxLayout()
        self.control_layout.setSpacing(10)
        
        # Name input
        self.name_label = QLabel("Enter name:")
        self.name_label.setStyleSheet(f"color: {WHITE};")
        self.name_input = QLineEdit()
        self.name_input.setStyleSheet(f"""
            background-color: #3A3000;
            border: 1px solid {PRIMARY_BLUE};
            color: {WHITE};
            padding: 8px;
        """)
        self.control_layout.addWidget(self.name_label)
        self.control_layout.addWidget(self.name_input)

        # First button row
        self.button_row1 = QHBoxLayout()
        self.button_row1.setSpacing(10)
        
        self.start_button = self.create_button("Start Camera", PRIMARY_BLUE)
        self.start_button.clicked.connect(self.start_camera)
        
        self.save_button = self.create_button("Save Face", "#00CC88", enabled=False)
        self.save_button.clicked.connect(self.save_face)
        
        self.button_row1.addWidget(self.start_button)
        self.button_row1.addWidget(self.save_button)
        self.control_layout.addLayout(self.button_row1)
        
                # زرار غلق الكاميرا
        self.stop_button = self.create_button("Stop Camera", "#DC143C", enabled=False)
        self.stop_button.clicked.connect(self.stop_camera)

        # زرار فتح ملف CSV
        self.open_csv_button = self.create_button("Open Attendance", "#4682B4")
        self.open_csv_button.clicked.connect(self.open_csv_file)

        self.button_row4 = QHBoxLayout()
        self.button_row4.setSpacing(10)
        self.button_row4.addWidget(self.stop_button)
        self.button_row4.addWidget(self.open_csv_button)
        self.control_layout.addLayout(self.button_row4)


        # Second button row
        self.button_row2 = QHBoxLayout()
        self.button_row2.setSpacing(10)
        
        self.mark_attendance_button = self.create_button("Mark Attendance", "#DAA520")
        self.mark_attendance_button.clicked.connect(self.mark_attendance)
        
        self.show_faces_button = self.create_button("Show Records", "#9C27B0")
        self.show_faces_button.clicked.connect(self.show_saved_faces)
        
        self.button_row2.addWidget(self.mark_attendance_button)
        self.button_row2.addWidget(self.show_faces_button)
        self.control_layout.addLayout(self.button_row2)

        # Third button row (admin only)
        self.button_row3 = QHBoxLayout()
        self.button_row3.setSpacing(10)
        
        self.delete_button = self.create_button("Delete User", ACCENT_RED, enabled=True)
        self.delete_button.clicked.connect(self.delete_user)
        
        self.edit_button = self.create_button("Edit User", "#FF9800", enabled=True)
        self.edit_button.clicked.connect(self.edit_user)
        
        self.button_row3.addWidget(self.delete_button)
        self.button_row3.addWidget(self.edit_button)
        self.control_layout.addLayout(self.button_row3)

        self.control_group.setLayout(self.control_layout)
        self.left_layout.addWidget(self.control_group)

    def setup_faces_list(self):
        """Initialize saved faces list"""
        self.faces_group = QGroupBox("Saved Faces")
        self.faces_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {PRIMARY_BLUE};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: {PRIMARY_BLUE};
                font-weight: bold;
                font-family: 'Poetsen One';
                font-size: 14pt;
            }}
        """)
        self.faces_layout = QVBoxLayout()
        
        self.faces_list = QListWidget()
        self.faces_list.itemClicked.connect(self.display_face)
        self.faces_list.setStyleSheet(f"""
            background-color: #3A3000;
            color: {WHITE};
            border: 1px solid {PRIMARY_BLUE};
            font-size: 14px;
            font-family: 'Playfair Display';
                font-size: 14pt;
        """)
        self.faces_layout.addWidget(self.faces_list)
        
        self.faces_group.setLayout(self.faces_layout)
        self.left_layout.addWidget(self.faces_group)

    def setup_right_panel(self):
        """Initialize right panel"""
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(10, 10, 10, 10)
        self.right_layout.setSpacing(15)

        # Camera group
        self.camera_group = QGroupBox("Camera")
        self.camera_group.setStyleSheet(f"""
            QGroupBox {{
                border: 1px solid {PRIMARY_BLUE};
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: {PRIMARY_BLUE};
                font-weight: bold;
                font-family: 'Poetsen One';
                font-size: 14pt;
            }}
        """)
        self.camera_layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(f"""
            border: 2px solid {BORDER_COLOR};
            border-radius: 8px;
            background-color: black;
            font-family: 'Poetsen One';
                font-size: 14pt;
        """)
        self.camera_layout.addWidget(self.video_label)
        
        self.recognition_status = QLabel("Camera not started")
        self.recognition_status.setAlignment(Qt.AlignCenter)
        self.recognition_status.setStyleSheet(f"""
            font-size: 16px;
            font-weight: bold;
            color: {PRIMARY_BLUE};
        """)
        self.camera_layout.addWidget(self.recognition_status)
        
        self.camera_group.setLayout(self.camera_layout)
        self.right_layout.addWidget(self.camera_group)

        self.right_layout.addStretch()

    def create_button(self, text, color, enabled=True):
        """Create a styled button"""
        button = QPushButton(text)
        button.setStyleSheet(f"""
            background-color: {color};
            color: white;
            border-radius: 6px;
            padding: 10px 15px;
            font-size: 16px;
            font-weight: bold;
            min-width: 120px;
        """)
        button.setEnabled(enabled)
        return button

    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.save_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.edit_button.setEnabled(True)

        self.recognition_status.setText("Searching for faces...")
        
    
    def stop_camera(self):
        """Stop the camera capture"""
        if self.cap and self.cap.isOpened():
            self.timer.stop()     # نوقف التايمر
            self.cap.release()    # نحرر الكاميرا
            self.cap = None       # نعمل cap = None لتفادي أي مشاكل بعدين
            self.video_label.clear()   # نمسح شاشة الفيديو
            self.recognition_status.setText("Camera stopped.")
            # self.save_button.setEnabled(False)
            # self.stop_button.setEnabled(False)


    def open_csv_file(self):
        """Open today's attendance CSV file"""
        try:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            filename = f"attendance_{today}.csv"
            directory = "attendance"
            filepath = os.path.join(os.getcwd(), directory, filename)

            if not os.path.exists(filepath):
                QMessageBox.warning(self, "Error", "No attendance file found for today!")
                return
            
            if platform.system() == "Windows":
                os.startfile(filepath)
            elif platform.system() == "Darwin":
                subprocess.call(["open", filepath])
            else:
                subprocess.call(["xdg-open", filepath])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")



    def update_frame(self):
        """Update camera frame"""
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb_frame)
        
        if len(faces) > 0:
            face = faces[0]
            self.current_face_image = frame[face.top():face.bottom(), face.left():face.right()]
            
            shape = shape_predictor(rgb_frame, face)
            self.current_face_descriptor = np.array(face_recognizer.compute_face_descriptor(rgb_frame, shape))
            
            match_name = self.find_face_match(self.current_face_descriptor)
            
            if match_name:
                self.last_recognized_name = match_name
                self.recognition_status.setText(f"Recognized: {match_name}")
                cv2.putText(frame, match_name, (face.left(), face.top()-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            else:
                self.last_recognized_name = None
                self.recognition_status.setText("New face - Click Save to add")
            
            cv2.rectangle(frame, (face.left(), face.top()), 
                          (face.right(), face.bottom()), (0,255,0), 2)
        else:
            self.last_recognized_name = None
            self.recognition_status.setText("Searching for faces...")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def find_face_match(self, descriptor, threshold=0.6):
        """Find matching face in database"""
        self.cursor.execute("SELECT name, descriptor FROM users")
        for name, saved_desc in self.cursor.fetchall():
            distance = np.linalg.norm(descriptor - np.frombuffer(saved_desc, dtype=np.float64))
            if distance < threshold:
                return name
        return None

    def save_face(self):
        """Save detected face to database"""
        name = self.name_input.text()
        if not name:
            name = "Unknown"

        if self.current_face_image is not None:
            try:
                shape = shape_predictor(
                    cv2.cvtColor(self.current_face_image, cv2.COLOR_BGR2RGB), 
                    detector(cv2.cvtColor(self.current_face_image, cv2.COLOR_BGR2RGB))[0]
                )
                descriptor = np.array(face_recognizer.compute_face_descriptor(
                    cv2.cvtColor(self.current_face_image, cv2.COLOR_BGR2RGB), 
                    shape
                ))
                
                _, img_bytes = cv2.imencode('.jpg', self.current_face_image)
                
                self.cursor.execute(
                    "INSERT INTO users (name, descriptor, image, created_at) VALUES (?, ?, ?, ?)",
                    (name, descriptor.tobytes(), img_bytes.tobytes(), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                self.conn.commit()
                QMessageBox.information(self, "Success", f"Face for {name} saved successfully!")
                self.show_saved_faces()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error while saving: {str(e)}")
        else:
            QMessageBox.warning(self, "Error", "No face detected!")

    def mark_attendance(self):
        """Mark attendance for recognized face"""
        if not self.last_recognized_name:
            QMessageBox.warning(self, "Error", "No recognized face to mark attendance!")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Create attendance directory if not exists
        if not os.path.exists("attendance"):
            os.makedirs("attendance")
            
        # Create or append to CSV file
        filename = f"attendance/attendance_{date}.csv"
        file_exists = os.path.isfile(filename)
        
        try:
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Name", "Timestamp"])
                writer.writerow([self.last_recognized_name, timestamp])
                
            QMessageBox.information(self, "Success", f"Attendance marked for {self.last_recognized_name} at {timestamp}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to mark attendance: {str(e)}")

    def show_saved_faces(self):
        """Display saved faces from database"""
        self.faces_list.clear()
        self.cursor.execute("SELECT name, image, created_at FROM users")
        rows = self.cursor.fetchall()
        for row in rows:
            name, img_data, created_at = row
            item = QListWidgetItem(f"{name} - {created_at}")
            item.setData(Qt.UserRole, img_data)
            self.faces_list.addItem(item)

    def display_face(self, item):
        """Display selected face image"""
        try:
            name = item.text().split(' - ')[0]
            self.cursor.execute("SELECT image FROM users WHERE name = ?", (name,))
            img_data = self.cursor.fetchone()[0]
            
            nparr = np.frombuffer(img_data, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            temp_filename = "temp_face.jpg"
            cv2.imwrite(temp_filename, img_np)
            
            if platform.system() == "Windows":
                os.startfile(temp_filename)
            elif platform.system() == "Darwin":
                subprocess.call(["open", temp_filename])
            else:
                subprocess.call(["xdg-open", temp_filename])
                
            self.selected_user = name
            self.delete_button.setEnabled(self.current_user["role"] == "admin")
            self.edit_button.setEnabled(self.current_user["role"] == "admin")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display image: {str(e)}")

    def delete_user(self):
        """Delete selected user from database"""
        if not self.selected_user:
            QMessageBox.warning(self, "Error", "No user selected!")
            return
            
        # if self.current_user["role"] != "admin":
        #     QMessageBox.warning(self, "Error", "You don't have permission to delete users!")
        #     return

        reply = QMessageBox.question(self, 'Confirm Delete', 
                                   f'Are you sure you want to delete {self.selected_user}?',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.cursor.execute("DELETE FROM users WHERE name = ?", (self.selected_user,))
            self.conn.commit()
            self.show_saved_faces()
            QMessageBox.information(self, "Success", f"User {self.selected_user} deleted!")
            self.selected_user = None

    def edit_user(self):
        """Edit selected user's name"""
        if not self.selected_user:
            QMessageBox.warning(self, "Error", "No user selected!")
            return
            
        # if self.current_user["role"] != "admin":
        #     QMessageBox.warning(self, "Error", "You don't have permission to edit users!")
        #     return

        new_name, ok = QInputDialog.getText(self, "Edit User", "Enter new name:", 
                                          QLineEdit.Normal, self.selected_user)
        if ok and new_name:
            self.cursor.execute("UPDATE users SET name = ? WHERE name = ?", 
                              (new_name, self.selected_user))
            self.conn.commit()
            self.show_saved_faces()
            QMessageBox.information(self, "Success", 
                                 f"User {self.selected_user} updated to {new_name}!")
            self.selected_user = None

    def closeEvent(self, event):
        """Clean up resources on window close"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        self.conn.close()
        event.accept()

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login - Face Recognition")
        self.setGeometry(100, 100, 500, 400)
        
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {DARK_BLUE};
                color: {WHITE};
            }}
            QLabel {{
                font-size: 16px;
                color: {PRIMARY_BLUE};
            }}
            QLineEdit, QComboBox {{
                background-color: #3A3000;
                border: 1px solid {PRIMARY_BLUE};
                border-radius: 4px;
                padding: 8px;
                color: {WHITE};
                font-size: 14px;
                min-width: 200px;
            }}
            QPushButton {{
                border-radius: 6px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
                min-width: 120px;
            }}
        """)

        self.layout = QVBoxLayout()
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(40, 40, 40, 40)

        self.logo_label = QLabel("Face Recognition")
        self.logo_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {PRIMARY_BLUE};
        """)
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.logo_label)

        self.form_layout = QVBoxLayout()
        self.form_layout.setSpacing(15)

        self.username_label = QLabel("Username:")
        self.username_input = QLineEdit()
        self.form_layout.addWidget(self.username_label)
        self.form_layout.addWidget(self.username_input)

        self.password_label = QLabel("Password:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.form_layout.addWidget(self.password_label)
        self.form_layout.addWidget(self.password_input)

        self.role_label = QLabel("Role:")
        self.role_combo = QComboBox()
        self.role_combo.addItem("user")
        self.role_combo.addItem("admin")
        self.form_layout.addWidget(self.role_label)
        self.form_layout.addWidget(self.role_combo)

        self.layout.addLayout(self.form_layout)

        self.button_layout = QVBoxLayout()
        self.button_layout.setSpacing(10)

        self.login_button = QPushButton("Login")
        self.login_button.setStyleSheet(f"""
            background-color: {PRIMARY_BLUE};
            color: white;
        """)

        self.signup_button = QPushButton("Sign Up")
        self.signup_button.setStyleSheet(f"""
            background-color: #00CC88;
            color: white;
        """)

        self.button_layout.addWidget(self.login_button)
        self.button_layout.addWidget(self.signup_button)
        self.layout.addLayout(self.button_layout)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

        self.login_button.clicked.connect(self.login)
        self.signup_button.clicked.connect(self.signup)

    def login(self):
        """Handle login process"""
        try:
            username = self.username_input.text()
            password = self.password_input.text()

            if not username or not password:
                QMessageBox.warning(self, "Error", "Please enter both username and password")
                return

            conn = None
            try:
                conn = create_connection()
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM accounts WHERE username = ? AND password = ?", (username, password))
                result = cursor.fetchone()
                
                if result:
                    self.current_user = {"username": username, "role": result[2]}
                    QMessageBox.information(self, "Success", "Login successful!")
                    self.main_window = MainWindow(self.current_user)
                    self.main_window.show()
                    self.close()
                else:
                    QMessageBox.warning(self, "Error", "Invalid username or password!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Database error: {e}")
            finally:
                if conn:
                    conn.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {e}")

    def signup(self):
        """Handle signup process"""
        try:
            username = self.username_input.text()
            password = self.password_input.text()
            role = self.role_combo.currentText()

            if not username or not password:
                QMessageBox.warning(self, "Error", "Please fill all fields!")
                return

            conn = None
            try:
                conn = create_connection()
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM accounts WHERE username = ?", (username,))
                result = cursor.fetchone()

                if result:
                    QMessageBox.warning(self, "Error", "Username already exists!")
                else:
                    cursor.execute("INSERT INTO accounts (username, password, role) VALUES (?, ?, ?)", 
                                (username, password, role))
                    conn.commit()
                    QMessageBox.information(self, "Success", "Account created successfully!")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Database error: {e}")
            finally:
                if conn:
                    conn.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {e}")   
            
@atexit.register
def cleanup():
    """Clean up temporary files on exit"""
    if os.path.exists("temp_face.jpg"):
        os.remove("temp_face.jpg")          
        
if __name__ == "__main__":

    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())