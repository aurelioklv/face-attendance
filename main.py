import datetime
import os
import subprocess
import tkinter as tk
import tkinter.filedialog

import cv2
from PIL import Image, ImageTk

import util



import tkinter as tk
from ultralytics import YOLO
from keras.models import Model
import math
import keras.backend as K
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Reshape,
    Bidirectional,
    LSTM,
    Dense,
    Lambda,
    Activation,
    BatchNormalization,
    Dropout,
    ReLU,
    GRU,
    Dot,
    Concatenate,
    RepeatVector,
    Permute,
    Multiply,
    Flatten,
)

import numpy as np
import time

model = YOLO("best.pt")

classNames = [
    "0",
    "0 0.3629690833333334 0.2073046875 0.255 0.194375",
    "face",
    "plate",
]

SHAPE = (128, 32)
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"



def adjust_brightness(image, gamma):
    # Apply gamma correction to adjust brightness
    adjusted_image = np.power(image / 255.0, gamma)
    adjusted_image = (adjusted_image * 255).astype(np.uint8)

    return adjusted_image


def adaptive_histogram_equalization(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(gray)

    # Convert back to BGR color space
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

    return equalized_image


def unsharp_masking(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)

    # Perform unsharp masking
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    # Convert back to BGR color space
    sharpened_image = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return sharpened_image


def preprocess_image(image):
    # Increase brightness using gamma correction
    # resize image

    height, width = image.shape[:2]

    if width < 400:
        scale = 400 / width
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    gamma = 0.8
    image = adjust_brightness(image, gamma)

    # Apply adaptive histogram equalization for contrast enhancement
    image = adaptive_histogram_equalization(image)

    # Apply unsharp masking for sharpening
    image = unsharp_masking(image)

    # Apply median blur for noise reduction
    image = cv2.medianBlur(image, 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    resized_image = cv2.resize(thresh, SHAPE)

    return resized_image


def words_from_labels(labels):
    """
    converts the list of encoded integer labels to word strings like eg. [12,10,29] returns CAT
    """
    txt = []
    for ele in labels:
        if ele == len(letters):  # CTC blank space
            txt.append("")
        elif ele == -1:
            txt.append("")
        else:
            # print(letters[ele])
            txt.append(letters[ele])
    return "".join(txt)


def test_data_single_image_Prediction(model, test_img_path):
    test_img = preprocess_image(test_img_path)
    img = Image.fromarray(test_img).convert("L").resize(SHAPE)
    img_array = np.array(img)
    img_array = img_array / 255.0

    # Tambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)  # Tambahkan dimensi saluran

    model_output = model.predict(img_array)

    predicted_output = K.get_value(
        K.ctc_decode(
            model_output,
            input_length=np.ones(model_output.shape[0]) * model_output.shape[1],
            greedy=True,
        )[0][0]
    )


    # get x, y coordinates from image
    org = (0, 25)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    decoded = words_from_labels(predicted_output[0])

    cv2.putText(test_img_path, decoded, org, font, fontScale, color, thickness)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"cropped_image_{timestamp}.jpg"
    cv2.imwrite(filename, test_img_path)
    print(f"Saved {filename}")


input_data = Input(shape=(SHAPE[1], SHAPE[0], 1), name="input")

inner = Conv2D(
    64, (3, 3), padding="same", name="conv1", kernel_initializer="he_normal"
)(input_data)
inner = BatchNormalization()(inner)
inner = Activation("relu")(inner)
inner = MaxPooling2D(pool_size=(2, 2), strides=2, name="max1")(inner)

inner = Conv2D(
    128, (3, 3), padding="same", name="conv2", kernel_initializer="he_normal"
)(inner)
inner = BatchNormalization()(inner)
inner = Activation("relu")(inner)
inner = MaxPooling2D(pool_size=(2, 2), strides=2, name="max2")(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(
    256, (3, 3), padding="same", name="conv3", kernel_initializer="he_normal"
)(inner)
inner = BatchNormalization()(inner)
inner = Activation("relu")(inner)
inner = MaxPooling2D(pool_size=(2, 1), name="max3")(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(
    256, (3, 3), padding="same", name="conv4", kernel_initializer="he_normal"
)(inner)
inner = BatchNormalization()(inner)
inner = Activation("relu")(inner)
inner = MaxPooling2D(pool_size=(2, 1), name="max4")(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(
    512, (3, 3), padding="same", name="conv5", kernel_initializer="he_normal"
)(inner)
inner = BatchNormalization()(inner)
inner = Activation("relu")(inner)
inner = MaxPooling2D(pool_size=(2, 1), name="max5")(inner)
inner = Dropout(0.3)(inner)

inner = Conv2D(
    512, (3, 3), padding="same", name="conv6", kernel_initializer="he_normal"
)(inner)
inner = BatchNormalization()(inner)
inner = Activation("relu")(inner)
inner = Dropout(0.3)(inner)

# CNN to RNN
inner = Reshape(target_shape=((32, 512)), name="reshape")(inner)
inner = Dense(128, activation="relu", kernel_initializer="he_normal", name="dense1")(
    inner
)

# RNN
inner = Bidirectional(LSTM(256, return_sequences=True), name="lstm1")(inner)
inner = Bidirectional(LSTM(256, return_sequences=True), name="lstm2")(inner)

# OUTPUT
inner = Dense(37, kernel_initializer="he_normal", name="dense2")(inner)
y_pred = Activation("softmax", name="softmax")(inner)


model_crnn = Model(inputs=input_data, outputs=y_pred)
model_crnn.load_weights("train_result.h5")




class App:
    def __init__(self):
        self.app_name = "Face Attendance System"
        self.video_capture_idx = 0
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
        print("detecting plate in image", image_path);
        img = cv2.imread(image_path)
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # crop image based on cv2.rectangle

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # class name
                cls = int(box.cls[0])

                # add code to crop image when key c is pressed
                if int(box.cls[0]) == 3:
                    crop_img = img[y1:y2, x1:x2]
                    test_data_single_image_Prediction(model_crnn, crop_img)


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

        # while True:
            # success, img = cap.read()
        ret, frame = self.cap.read()
        results = model(frame, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # put box in cam
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # crop image based on cv2.rectangle

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                # print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                # print("Class name -->", classNames[cls])

                # add code to crop image when key c is pressed
                # if keyboard.is_pressed('c'):
                # if int(box.cls[0]) == 3:
                #     crop_img = img[y1:y2, x1:x2]
                #     test_data_single_image_Prediction(model_crnn, crop_img)


                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)


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
