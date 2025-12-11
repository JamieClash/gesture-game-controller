import json
import pydirectinput
import tkinter as tk
from tkinter import ttk

SUPPORTED_MOUSE_EVENTS = ["left_click", "right_click", "middle_click", "double_click", "triple_click"]
SUPPORTED_KEYS = sorted(pydirectinput.KEYBOARD_MAPPING.keys())
NULL_KEY_SELECTION = ["."]

class ScrollWidget(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # canvas for scrolling
        self.canvas = tk.Canvas(self, borderwidth=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.h_scroll = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.h_scroll.pack(side="bottom", fill="x")
        self.v_scroll = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.v_scroll.pack(side="right", fill="y")

        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.inner = tk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # update size of scroll region
        self.inner.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # mousewheel scrolling
        self.inner.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self.on_scroll))
        self.inner.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        self.canvas.itemconfig(self.window_id, width=event.width)

    def on_scroll(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

def load_profile(path):
    with open(path, "r") as f:
        return json.load(f)

def save_profile(profile, path):
    with open(path, "w") as f:
        json.dump(profile, f, indent=4)

def open_gui(path, root):
    profile = load_profile(path)
    profile_name = profile["profile_name"]

    window = tk.Toplevel(root)
    window.title(f"Editing {profile_name}")

    canvas = ScrollWidget(window)
    canvas.pack(fill="both", expand=True)

    row = 0
    for hand in ["left", "right"]:
        tk.Label(canvas.inner, text=f"{hand.upper()} HAND", font=("Arial", 14, "bold")).grid(row=row, column=0, sticky="w", pady=10)
        row += 1

        for gesture_id, settings in profile["gestures"][hand].items():
            gesture_name = settings["name"]
            tk.Label(canvas.inner, text=f"[{gesture_name}]").grid(row=row, column=0, sticky="w")

            # dropdown for enable/disable
            enable_var = tk.BooleanVar(value=bool(settings["enable"]))
            enable_box = ttk.Checkbutton(canvas.inner, variable=enable_var)
            enable_box.grid(row=row, column=1)

            # gesture hand state is NOT configurable

            # dropdown for action type
            action_var = tk.StringVar(value=settings["action"])
            action_box = ttk.Combobox(canvas.inner, textvariable=action_var, values=["key", "mouse_move", "mouse_click", "none"])
            action_box.grid(row=row, column=2)

            # Dropdown for key
            key_var = tk.StringVar(value=settings["key"])
            actionType = settings["action"]
            if actionType == "key":
                key_box = ttk.Combobox(canvas.inner, textvariable=key_var, values=SUPPORTED_KEYS)
            elif actionType == "mouse_click":
                key_box = ttk.Combobox(canvas.inner, textvariable=key_var, values=SUPPORTED_MOUSE_EVENTS)
            else:
                key_box = ttk.Combobox(canvas.inner, textvariable=key_var, values=NULL_KEY_SELECTION)
            key_box.grid(row=row, column=3)

            # Update profile and save
            def save_changes(event=None, hand=hand, gesture_id=gesture_id, enable_var=enable_var,
                             action_var=action_var, key_var=key_var):
                profile["gestures"][hand][gesture_id]["enable"] = int(enable_var.get())
                profile["gestures"][hand][gesture_id]["action"] = action_var.get()
                profile["gestures"][hand][gesture_id]["key"] = key_var.get()
                # ...

                # transition functions too
                # ...

                save_profile(profile, path)

            # Save when dropdown changes
            enable_box.configure(command=save_changes)
            action_box.bind("<<ComboboxSelected>>", save_changes)
            key_box.bind("<<ComboboxSelected>>", save_changes)

            row += 1
    
    # transition function settings go here
    # ...

    # reveal hidden root when closing mapping GUI
    def on_close():
        window.destroy()
        root.deiconify()
    
    window.protocol("WM_DELETE_WINDOW", on_close)

    return window
