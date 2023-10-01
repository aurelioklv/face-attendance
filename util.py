import tkinter as tk
from tkinter import messagebox

import face_recognition


def get_button(window, text, color, command, foreground="white"):
    button = tk.Button(
        window,
        text=text,
        bg=color,
        fg=foreground,
        command=command,
        activebackground="black",
        activeforeground="white",
        font=("Helvetica", 20, "bold"),
        height=2,
        width=20,
    )
    return button


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_input_text(window):
    input_text = tk.Text(window, height=2, width=15, font=("Arial", 32))
    return input_text


def msg_box(title, description):
    messagebox.showinfo(title, description)
