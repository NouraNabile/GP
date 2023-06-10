import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from seg import *

case_number = '001'

def process_number():
    # Get the number from the input field
    number = number_input.get()
    if len(number) != 3:
        messagebox.showerror("Error", "Please enter a 3-digit number.")
    elif int(number) > 369 or int(number) < 1:
        messagebox.showerror("Error", "This case is not available!")
    else:
        global case_number
        case_number = number

        print(number)

def function1():
    # Add your code for function 1 here
    Show3DTumor(case_number)

def function2():
    # Add your code for function 2 here
    show3Dbrain(case_number)

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
case_number_label = tk.Label(main_window, text="Enter case number in 3 digits between '001 and '369':",  fg="#eebad4", font=("Helvetica", 24, "bold"))
case_number_label.place(relx=0.5, rely=0.4, anchor="center")
number_input = tk.Entry(main_window)
number_input.place(relx=0.5, rely=0.45, anchor="center")
number_input.config(width=30, font=("Helvetica", 20))

# Create the button to submit the number
submit_button = tk.Button(main_window, text="Submit", command=process_number, bg="#ddd8da", fg="white",font=("Helvetica", 20, "bold"))
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