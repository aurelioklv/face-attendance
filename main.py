import datetime
import os
import subprocess
import tkinter as tk
import tkinter.filedialog

import cv2
from PIL import Image, ImageTk

import util


class App:
    def __init__(self):
        self.app_name = "Face Attendance System"
        self.video_capture_idx = "/dev/video0"
        self.main_window = tk.Tk()
        self.main_window.geometry("1100x520")
        self.main_window.resizable(False, False)
        self.main_window.title(self.app_name)

        self.db_dir = "./known_faces"
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = "./log.txt"

        self.login_button_main_window = util.get_button(
            self.main_window, "Login", "green", self.login
        )
        self.login_button_main_window.place(x=750, y=300)

        self.register_button_main_window = util.get_button(
            self.main_window, "Register", "grey", self.register, foreground="black"
        )
        self.register_button_main_window.place(x=750, y=400)

        self.select_face_image_button = util.get_button(
            self.main_window, "Select Face Image", "blue", self.select_face_image
        )
        self.select_face_image_button.place(x=750, y=200)

        self.select_face_image_button = util.get_button(
            self.main_window, "Select Plate Image", "blue", self.select_plate_image
        )
        self.select_face_image_button.place(x=750, y=100)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

    def start(self):
        self.main_window.mainloop()

    def login(self):
        unknown_image_path = "./.tmp.jpg"
        cv2.imwrite(unknown_image_path, self.most_recent_capture_arr)

        try:
            # Run face_recognition command and use cut to extract the second field
            output = subprocess.check_output(
                f"face_recognition {self.db_dir} {unknown_image_path} --tolerance 0.4 | cut -d ',' -f2",
                shell=True,
                text=True,  # Decode output to string
            )

            name = output.strip()
            if name in ["no_persons_found", "unknown_person"]:
                self.log("UNDETECTED" if name == "no_persons_found" else "UNKNOWN")
                util.msg_box("Oops", "Please register or try again!")
            else:
                self.log("LOGIN: {}".format(name))
                util.msg_box("Success", "Welcome, {}!".format(name))

        except subprocess.CalledProcessError as e:
            self.log(f"Error: {e}")
            util.msg_box("Error", "An error occurred. Please try again.")

        # Remove the temporary image file
        os.remove(unknown_image_path)

    def register(self):
        self.register_window = tk.Toplevel(self.main_window)
        self.register_window.geometry("1100x520")
        self.register_window.resizable(False, False)
        self.register_window.title(self.app_name + " - Register")

        self.text_label_register_window = util.get_text_label(
            self.register_window, "Name:"
        )
        self.text_label_register_window.place(x=750, y=100)

        self.entry_text_register_window = util.get_input_text(self.register_window)
        self.entry_text_register_window.place(x=750, y=150)

        self.accept_button_register_window = util.get_button(
            self.register_window, "Accept", "green", self.accept_register
        )
        self.accept_button_register_window.place(x=750, y=300)

        self.retake_button_register_window = util.get_button(
            self.register_window, "Retake Photo", "red", self.retake_register
        )
        self.retake_button_register_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

    def select_face_image(self):
        file_path = tk.filedialog.askopenfilename(title="Select Image File")
        if file_path:
            self.detect_face_in_image(file_path)

    def select_plate_image(self):
        file_path = tk.filedialog.askopenfilename(title="Select Plate File")
        if file_path:
            self.detect_plate_in_image(file_path)

    def detect_face_in_image(self, image_path):
        output = str(
            subprocess.check_output(["face_recognition", self.db_dir, image_path])
        )
        name = output.split(",")[1][:-3]
        if name in ["no_persons_found", "unknown_person"]:
            self.log("UNDETECTED" if name == "no_persons_found" else "UNKNOWN")
            util.msg_box("oops", "Please register or try again!")
        else:
            self.log("LOGIN:{}".format(name))
            util.msg_box("Success", "Welcome, {}!".format(name))

    def detect_plate_in_image(self, image_path):
        pass

    def accept_register(self):
        name = self.entry_text_register_window.get("1.0", "end-1c")
        cv2.imwrite(
            os.path.join(self.db_dir, "{}.jpg".format(name)),
            self.registered_new_user_capture,
        )
        self.log("REGISTER:{}".format(name))
        util.msg_box("Success", "User registered successfully!")
        self.register_window.destroy()

    def retake_register(self):
        self.register_window.destroy()

    def add_img_to_label(self, label):
        imgTk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgTk = imgTk
        label.configure(image=imgTk)

        self.registered_new_user_capture = self.most_recent_capture_arr.copy()

    def add_webcam(self, label):
        if "cap" not in self.__dict__:
            self.cap = cv2.VideoCapture(self.video_capture_idx)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        self.most_recent_capture_arr = frame

        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)

        imgTk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgTk = imgTk
        self._label.configure(image=imgTk)

        self._label.after(10, self.process_webcam)

    def log(self, text):
        with open(self.log_path, "a") as f:
            f.write("{} {}\n".format(datetime.datetime.now(), text))


if __name__ == "__main__":
    app = App()
    app.start()
