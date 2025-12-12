import tkinter as tk
from tkinter import filedialog
import sys

def select_profile():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Gesture Mapping Profile",
        filetypes=[("JSON files", "*.json")]
    )

    if file_path:
        print(file_path)
    else:
        print("")

    sys.exit(0)

if __name__ == "__main__":
    select_profile()
