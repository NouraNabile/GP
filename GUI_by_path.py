import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from seg_by_path import *
from tkinter import filedialog
path=""
def select_folder():
    # Show the file dialog and get the selected folder
    folder_path = filedialog.askdirectory()
    # Update the text in the label with the selected folder
    folder_label.config(text=folder_path)
    global path
    path=folder_path
    print(path)



def function1():
    global path
    Show3DTumor(path)

def function2():
    global path
    show3Dbrain(path)

# Set the background image
bg_image = Image.open("back2.png")
bg_width, bg_height = bg_image.size

# Create the main main_window
main_window = tk.Tk()
main_window.title("My App")

# Set the main_window size to match the background image
main_window.geometry(f"{bg_width}x{bg_height}")

# Create a label with the background image
bg_photo = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(main_window, image=bg_photo)
bg_label.place(x=0, y=0)


# Create the input field for the number
case_number_label = tk.Label(main_window, text="enter the path of your case",  fg="#eebad4", font=("Helvetica", 24, "bold"))
case_number_label.place(relx=0.5, rely=0.4, anchor="center")
folder_label = tk.Label(main_window, text="No folder selected")
folder_label.place(relx=0.5, rely=0.45, anchor="center")
folder_label.config(width=30, font=("Helvetica", 20))

# Create the button to submit the number
submit_button = tk.Button(main_window, text="Submit", command=select_folder, bg="#ddd8da", fg="white",font=("Helvetica", 20, "bold"))
submit_button.place(relx=0.5, rely=0.5, anchor="center")

# Create the button to call function 1
function1_button = tk.Button(main_window, text="view 3D tumor", command=function1, bg="#eebad4", fg="white", font=("Helvetica", 20, "bold"))
function1_button.place(relx=0.5, rely=0.6, anchor="center")

# Create the button to call function 2
function2_button = tk.Button(main_window, text="View 3D brain with tumor", command=function2, bg="#a86bf1", fg="white", font=("Helvetica", 20, "bold"))
function2_button.place(relx=0.5, rely=0.7, anchor="center")

# Create a label to display the results
result_label = tk.Label(main_window, text="", bg="white")
result_label.pack()

# Start the main event loop
main_window.mainloop()