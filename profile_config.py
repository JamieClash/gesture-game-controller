import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import json
import os
from GUIs.mapping_GUI import open_gui

profile_dir = "./profiles/custom_profiles/"
     
def create_new_profile(directory=profile_dir):
    if not os.path.exists(directory):
        os.makedirs(directory)

    name = simpledialog.askstring("New Profile", "Please enter a name: ")
    filename = name

    if not filename:
        return None

    if not filename.endswith(".json"):
        filename += ".json"

    new_path = profile_dir + filename

    # prevent overwriting onto existing profile.
    if os.path.exists(new_path):
        messagebox.showerror("Error", "A profile with that name already exists.")
        return None

    # extract default profile structure from corresponding file
    default_path = profile_dir + "default.json"
    with open(default_path, "r") as f:
        default_profile = json.load(f)
    
    # set profile name
    default_profile["profile_name"] = name

    with open(new_path, "w") as f:
        json.dump(default_profile, f, indent=4)

    return new_path


def choose_existing_profile():
    return filedialog.askopenfilename(
        title="Select Profile JSON",
        filetypes=[("JSON files", "*.json")]
    )


def main():
    root = tk.Tk()
    root.title("Profile Loader")
    root.geometry("300x180")

    tk.Label(root, text="Gesture Mapping Profiles", font=("Arial", 14)).pack(pady=10)

    def load_existing():
        path = choose_existing_profile()
        if path:
            root.withdraw()
            open_gui(path, root)

    def create_profile():
        path = create_new_profile()
        if path:
            root.withdraw()
            open_gui(path, root)

    tk.Button(root, text="Load Existing Profile", width=22, command=load_existing).pack(pady=5)
    tk.Button(root, text="Create New Profile", width=22, command=create_profile).pack(pady=5)
    tk.Button(root, text="Exit", width=22, command=root.quit).pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()