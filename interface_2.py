import numpy as np
import cv2
import mediapipe as mp
import time
import sqlite3
from tkinter import ttk, Label, Entry, Button, messagebox, Frame, Tk
from PIL import Image, ImageTk
from utils.utils_v2 import get_idx_to_coordinates, rescale_frame

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# Mock user and employee databases
users = {"user": "password"}
employees = {"E001": "Alice", "E002": "Bob", "E003": "Charlie"}

def log_activity(employee_id, activity_type, product_count=0):
    conn = sqlite3.connect('employee_tracking.db')
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO ActivityLog (employee_id, activity_type, product_count)
    VALUES (?, ?, ?)
    ''', (employee_id, activity_type, product_count))

    conn.commit()
    conn.close()

class HandTrackingApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x800")
        self.window.configure(bg="#2c3e50")

        self.login_screen()

    def login_screen(self):
        self.clear_window()

        self.login_frame = Frame(self.window, bg="#2c3e50")
        self.login_frame.pack(pady=150)

        self.username_label = Label(self.login_frame, text="Username:", font=("Helvetica", 14), bg="#2c3e50", fg="#ecf0f1")
        self.username_label.grid(row=0, column=0, padx=10, pady=5)
        self.username_entry = Entry(self.login_frame, font=("Helvetica", 14))
        self.username_entry.grid(row=0, column=1, padx=10, pady=5)

        self.password_label = Label(self.login_frame, text="Password:", font=("Helvetica", 14), bg="#2c3e50", fg="#ecf0f1")
        self.password_label.grid(row=1, column=0, padx=10, pady=5)
        self.password_entry = Entry(self.login_frame, font=("Helvetica", 14), show="*")
        self.password_entry.grid(row=1, column=1, padx=10, pady=5)

        self.login_button = Button(self.login_frame, text="Login", command=self.login, bg="#3498db", fg="#ecf0f1", font=("Helvetica", 14))
        self.login_button.grid(row=2, columnspan=2, pady=20)

    def clear_window(self):
        for widget in self.window.winfo_children():
            widget.destroy()

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if username in users and users[username] == password:
            self.employee_selection_screen()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password")

    def employee_selection_screen(self):
        self.clear_window()

        self.selection_frame = Frame(self.window, bg="#2c3e50")
        self.selection_frame.pack(fill='both', expand=True)

        self.tree = ttk.Treeview(self.selection_frame, columns=("ID", "Name"), show='headings', height=20)
        self.tree.heading("ID", text="Employee ID")
        self.tree.heading("Name", text="Employee Name")
        self.tree.column("ID", width=100)
        self.tree.column("Name", width=200)

        for emp_id, emp_name in employees.items():
            self.tree.insert("", "end", values=(emp_id, emp_name))

        self.tree.pack(fill='both', expand=True)

        self.select_button = Button(self.selection_frame, text="Start Tracking", command=self.select_employee, bg="#3498db", fg="#ecf0f1", font=("Helvetica", 14))
        self.select_button.pack(pady=20)

        self.report_button = Button(self.selection_frame, text="Generate Report", command=self.generate_report, bg="#3498db", fg="#ecf0f1", font=("Helvetica", 14))
        self.report_button.pack(pady=20)

    def select_employee(self):
        selected = self.tree.focus()
        if not selected:
            messagebox.showerror("Selection Error", "Please select an employee")
            return

        emp_info = self.tree.item(selected, 'values')
        self.emp_id, self.emp_name = emp_info
        print(f"Selected Employee ID: {self.emp_id}, Name: {self.emp_name}")
        self.main_screen()

    def generate_report(self):
        selected = self.tree.focus()
        if not selected:
            messagebox.showerror("Selection Error", "Please select an employee")
            return

        emp_info = self.tree.item(selected, 'values')
        emp_id, emp_name = emp_info
        self.create_report(emp_id, emp_name)

    def create_report(self, emp_id, emp_name):
        conn = sqlite3.connect('employee_tracking.db')
        cursor = conn.cursor()

        cursor.execute('''
        SELECT activity_type, SUM(product_count), COUNT(activity_type), SUM(working_time), SUM(idle_time)
        FROM ActivityLog
        WHERE employee_id = ?
        GROUP BY activity_type
        ''', (emp_id,))
        data = cursor.fetchall()

        report = f"Report for {emp_name} (ID: {emp_id}):\n\n"
        for row in data:
            report += f"Activity Type: {row[0]}\n"
            report += f"Total Product Count: {row[1]}\n"
            report += f"Activity Count: {row[2]}\n"
            report += f"Total Working Time: {row[3]} seconds\n"
            report += f"Total Idle Time: {row[4]} seconds\n\n"

        print(report)
        messagebox.showinfo("Employee Report", report)

        conn.close()

    def clear_window(self):
        for widget in self.window.winfo_children():
            widget.destroy()

    def main_screen(self):
        self.clear_window()

        self.video_frame = Frame(self.window, bg="#2c3e50")
        self.video_frame.pack(side="top", pady=20)

        self.video_label = Label(self.video_frame)
        self.video_label.pack()

        self.info_frame = Frame(self.window, bg="#2c3e50")
        self.info_frame.pack(side="top", pady=20)

        self.working_time_label = Label(self.info_frame, text="Working Time: 0.00s", font=("Helvetica", 14), bg="#2c3e50", fg="#ecf0f1")
        self.working_time_label.grid(row=0, column=0, padx=10, pady=5)
        self.idle_time_label = Label(self.info_frame, text="Idle Time: 0.00s", font=("Helvetica", 14), bg="#2c3e50", fg="#ecf0f1")
        self.idle_time_label.grid(row=0, column=1, padx=10, pady=5)
        self.product_count_label = Label(self.info_frame, text="Product Count: 0", font=("Helvetica", 14), bg="#2c3e50", fg="#ecf0f1")
        self.product_count_label.grid(row=0, column=2, padx=10, pady=5)

        self.limit_frame = Frame(self.window, bg="#2c3e50")
        self.limit_frame.pack(side="top", pady=20)

        self.idle_time_limit_label = Label(self.limit_frame, text="Set Idle Time Limit (H:M:S):", font=("Helvetica", 12), bg="#2c3e50", fg="#ecf0f1")
        self.idle_time_limit_label.grid(row=0, column=0, padx=5)
        self.hour_entry = Entry(self.limit_frame, width=5, font=("Helvetica", 12))
        self.hour_entry.grid(row=0, column=1, padx=5)
        self.minute_entry = Entry(self.limit_frame, width=5, font=("Helvetica", 12))
        self.minute_entry.grid(row=0, column=2, padx=5)
        self.second_entry = Entry(self.limit_frame, width=5, font=("Helvetica", 12))
        self.second_entry.grid(row=0, column=3, padx=5)
        self.set_limit_button = Button(self.limit_frame, text="Set Limit", command=self.set_idle_time_limit, bg="#3498db", fg="#ecf0f1", font=("Helvetica", 12))
        self.set_limit_button.grid(row=0, column=4, padx=5)

        self.logout_button = Button(self.window, text="Logout", command=self.logout, bg="#e74c3c", fg="#ecf0f1", font=("Helvetica", 12))
        self.logout_button.pack(pady=20)

        self.hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.cap = cv2.VideoCapture(0)

        self.count = 0
        self.hands_in_red_zone = False

        self.working_time = 0
        self.idle_time = 0
        self.last_update_time = time.time()
        self.last_movement_time = time.time()
        self.prev_hand_positions = []

        self.idle_time_limit = None
        self.idle_time_alert_shown = False

        self.update()

    def logout(self):
        self.clear_window()
        self.cap.release()
        self.login_screen()

    def set_idle_time_limit(self):
        try:
            hours = int(self.hour_entry.get())
            minutes = int(self.minute_entry.get())
            seconds = int(self.second_entry.get())
            self.idle_time_limit = hours * 3600 + minutes * 60 + seconds
            self.idle_time_alert_shown = False  # Reset the alert flag
            messagebox.showinfo("Success", f"Idle time limit set to {hours} hours, {minutes} minutes, and {seconds} seconds.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for hours, minutes, and seconds.")

    def define_zones(self, image, left_shoulder, right_shoulder):
        height, width, _ = image.shape
        body_width = abs(right_shoulder[0] - left_shoulder[0])
        body_height = height // 2

        gap = 20

        front_zone = (
            left_shoulder[0],
            left_shoulder[1],
            right_shoulder[0],
            left_shoulder[1] + body_height
        )

        right_zone = (
            right_shoulder[0] + body_width + gap,
            right_shoulder[1],
            right_shoulder[0] + 2 * body_width + gap,
            right_shoulder[1] + body_height
        )

        return front_zone, right_zone

    def is_in_zone(self, point, zone):
        x, y = point
        x1, y1, x2, y2 = zone
        return x1 < x < x2 and y1 < y < y2

    def update(self):
        ret, image = self.cap.read()
        if not ret:
            self.window.after(10, self.update)
            return

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hand = self.hands.process(image_rgb)
        results_pose = self.pose.process(image_rgb)
        image.flags.writeable = True

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]),
                             int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.shape[0]))
            right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]),
                              int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0]))

            front_zone, right_zone = self.define_zones(image, left_shoulder, right_shoulder)

            cv2.circle(image, left_shoulder, 5, (0, 255, 0), -1)
            cv2.circle(image, right_shoulder, 5, (0, 255, 0), -1)

            hands_positions = []
            if results_hand.multi_hand_landmarks:
                for hand_landmarks in results_hand.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=5, circle_radius=5),
                        connection_drawing_spec=mp_drawing.DrawingSpec(thickness=10, circle_radius=10))
                    idx_to_coordinates = get_idx_to_coordinates(image, results_hand)
                    if 8 in idx_to_coordinates:
                        hands_positions.append(idx_to_coordinates[8])

            both_hands_in_red_zone = False

            if len(hands_positions) >= 2:
                both_hands_in_red_zone = all(self.is_in_zone(hand_position, right_zone) for hand_position in hands_positions[:2])
                if both_hands_in_red_zone and not self.hands_in_red_zone:
                    self.count += 1
                    self.hands_in_red_zone = True
                    log_activity(self.emp_id, 'working', self.count)  # Log working activity
                elif not both_hands_in_red_zone:
                    self.hands_in_red_zone = False

            current_time = time.time()
            elapsed_time = current_time - self.last_update_time

            hand_movement = False
            if self.prev_hand_positions:
                hand_movement = any(np.linalg.norm(np.array(prev) - np.array(curr)) > 20 for prev, curr in zip(self.prev_hand_positions, hands_positions))

            if hand_movement or both_hands_in_red_zone:
                self.working_time += elapsed_time
                self.last_movement_time = current_time
                self.idle_time_alert_shown = False  # Reset alert flag if there's movement
            else:
                if current_time - self.last_movement_time > 2:
                    self.idle_time += elapsed_time
                    log_activity(self.emp_id, 'idle')  # Log idle activity

            self.prev_hand_positions = hands_positions if hands_positions else self.prev_hand_positions
            self.last_update_time = current_time

            if self.idle_time_limit and self.idle_time > self.idle_time_limit and not self.idle_time_alert_shown:
                self.idle_time_alert_shown = True
                messagebox.showwarning("Idle Time Alert", "Idle time has reached the set limit!")

            cv2.rectangle(image, front_zone[:2], front_zone[2:], (0, 255, 0), 2)
            cv2.rectangle(image, right_zone[:2], right_zone[2:], (0, 0, 255), 2)
            cv2.putText(image, f"Count: {self.count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Working Time: {self.working_time:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Idle Time: {self.idle_time:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=image)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.working_time_label.config(text=f"Working Time: {self.working_time:.2f}s")
        self.idle_time_label.config(text=f"Idle Time: {self.idle_time:.2f}s")
        self.product_count_label.config(text=f"Product Count: {self.count}")

        self.window.after(10, self.update)

if __name__ == "__main__":
    root = Tk()
    app = HandTrackingApp(root, "Hand Tracking Interface")
    root.mainloop()
