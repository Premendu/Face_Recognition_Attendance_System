import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import os
import threading

# ===== PATH-INDEPENDENT BASE =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(BASE_DIR, "classes")

running = False
video_capture = None

# ========= CHECK CLASS =========
def check_class_exists(class_name):
    class_path = os.path.join(BASE_PATH, class_name)
    if not os.path.exists(class_path):
        messagebox.showerror("Error", f"No class named '{class_name}' exists")
        return False
    return True


# ========= UPDATE ENCODINGS =========
def update_encodings():
    try:
        class_name = class_entry.get().strip()

        if not class_name:
            messagebox.showerror("Error","Enter class name")
            return

        if not check_class_exists(class_name):
            return

        excel_file = os.path.join(BASE_PATH, class_name, f"{class_name}.xlsx")

        df = pd.read_excel(excel_file)
        names = df["Name"].tolist()

        enc_list = []

        for i, name in enumerate(names, start=1):

            status_label.config(
                text=f"Updating encoding for student {i}: {name}..."
            )
            root.update_idletasks()

            img_path = os.path.join(BASE_PATH, class_name, "faces", f"{i}.jpg")

            if os.path.exists(img_path):
                img = face_recognition.load_image_file(img_path)
                enc = face_recognition.face_encodings(img)
                enc_list.append(enc[0] if enc else np.zeros(128))
            else:
                enc_list.append(np.zeros(128))

        enc_file = os.path.join(BASE_PATH, class_name, f"{class_name}_encodings.pkl")

        with open(enc_file, "wb") as f:
            pickle.dump((names, enc_list), f)

        status_label.config(text="✅ All encodings updated!", fg="green")

    except Exception as e:
        status_label.config(text=str(e), fg="red")


# ========= ATTENDANCE LOOP =========
def attendance_loop():
    global running, video_capture

    class_name = class_entry.get().strip()

    excel_file = os.path.join(BASE_PATH, class_name, f"{class_name}.xlsx")
    enc_file = os.path.join(BASE_PATH, class_name, f"{class_name}_encodings.pkl")

    df = pd.read_excel(excel_file)

    with open(enc_file, "rb") as f:
        names, encodings = pickle.load(f)

    students = names.copy()
    date = datetime.now().strftime("%d.%m.%Y")

    video_capture = cv2.VideoCapture(0)

    while running:
        ret, frame = video_capture.read()
        if not ret:
            break

        small = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        rgb = cv2.cvtColor(small,cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb,locs)

        locs=[(t*2,r*2,b*2,l*2) for (t,r,b,l) in locs]

        for enc,(top,right,bottom,left) in zip(encs,locs):
            dists = face_recognition.face_distance(encodings, enc)
            best = np.argmin(dists)

            if dists[best] < 0.4:
                name = names[best]

                cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
                cv2.putText(frame,name,(left,top-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

                if name in students:
                    students.remove(name)
                    df.at[best,date] = "P "+datetime.now().strftime("%H:%M:%S")
                    df.to_excel(excel_file,index=False)

        cv2.imshow("Attendance", frame)

        if cv2.waitKey(1)&0xFF == ord('q'):
            stop_program()

    stop_program()


# ========= START =========
def take_attendance():
    global running

    class_name = class_entry.get().strip()

    if not class_name:
        messagebox.showerror("Error","Enter class name")
        return

    if not check_class_exists(class_name):
        return

    running = True
    threading.Thread(target=attendance_loop, daemon=True).start()


# ========= STOP =========
def stop_program():
    global running, video_capture
    running = False

    if video_capture:
        video_capture.release()

    cv2.destroyAllWindows()


# ========= GUI =========
root = tk.Tk()
root.title("Face_Recognition_Attendance_System")
root.state("zoomed")
root.configure(bg="#4914cf")

tk.Label(root, text="Face Recognition Attendance System",
         font=("Arial",28,"bold"),
         bg="#4914cf", fg="white").pack(pady=40)

tk.Label(root, text="Enter Class Name:",
         font=("Arial",18),
         bg="#4914cf", fg="white").pack()

class_entry = tk.Entry(root, font=("Arial",18))
class_entry.pack(pady=15)

status_label = tk.Label(root, text="",
                        font=("Arial",16),
                        bg="#4914cf", fg="white")
status_label.pack(pady=20)

tk.Button(root, text="Update Face Encodings",
          command=update_encodings,
          width=25, height=2,
          font=("Arial",14,"bold"),
          bg="#007bff", fg="white").pack(pady=20)

tk.Button(root, text="Take Attendance",
          command=take_attendance,
          width=25, height=2,
          font=("Arial",14,"bold"),
          bg="#112e6b", fg="white").pack(pady=20)

tk.Button(root, text="Exit",
          command=lambda:[stop_program(), root.quit()],
          width=25, height=2,
          font=("Arial",14,"bold"),
          bg="#dc3545", fg="white").pack(pady=20)

root.mainloop()