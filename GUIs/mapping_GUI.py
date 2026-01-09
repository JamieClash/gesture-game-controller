import json
import pydirectinput
import tkinter as tk
from tkinter import ttk

HANDS = ["left", "right"]
FINGERS = ["thumb", "index", "middle", "fourth", "pinky"]

SUPPORTED_MOUSE_EVENTS = ["left_click", "right_click", "middle_click", "double_click", "scroll_wheel_up", "scroll_wheel_down"]
SUPPORTED_KEYS = ["NULL"] + sorted(pydirectinput.KEYBOARD_MAPPING.keys())

ACTION_MODES = ["key", "mouse_move", "mouse_click", "none"]
KEY_MODES = ["tap", "hold"]
CURSOR_MODES = ["none", "relative movement", "panning", "angle movement"]  # 0, 1, 2, 3 in json

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

        # mousewheel scrolling
        self.inner.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self.on_scroll))
        self.inner.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

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
    for hand in HANDS:
        tk.Label(canvas.inner, text=f"{hand.upper()} HAND GESTURES", font=("Arial", 14, "bold")).grid(row=row, column=0, sticky="w", pady=10)
        row += 1

        tk.Label(canvas.inner, text="Gesture").grid(row=row, column=0)
        tk.Label(canvas.inner, text="Enabled").grid(row=row, column=1)
        tk.Label(canvas.inner, text="Action").grid(row=row, column=2)
        tk.Label(canvas.inner, text="Key").grid(row=row, column=3)
        tk.Label(canvas.inner, text="Mode").grid(row=row, column=4)
        tk.Label(canvas.inner, text="Retrigger Action").grid(row=row, column=5)
        tk.Label(canvas.inner, text="Retrigger Key").grid(row=row, column=6)
        tk.Label(canvas.inner, text="Retrigger Mode").grid(row=row, column=7)
        tk.Label(canvas.inner, text="Retrigger Cooldown (ms)").grid(row=row, column=8)
        tk.Label(canvas.inner, text="Finger Transitions").grid(row=row, column=9)
        tk.Label(canvas.inner, text="Cursor Mode").grid(row=row, column=10)
        tk.Label(canvas.inner, text="Cursor Sensitivity").grid(row=row, column=11)
        row += 1

        for gesture_id, settings in profile["gestures"][hand].items():
            gesture_name = settings["name"]
            tk.Label(canvas.inner, text=f"{gesture_name}").grid(row=row, column=0, sticky="w")

            # checkbox for enable/disable
            enable_var = tk.BooleanVar(value=bool(settings["enabled"]))
            enable_box = ttk.Checkbutton(canvas.inner, variable=enable_var)
            enable_box.grid(row=row, column=1)

            # gesture hand state is NOT configurable

            # dropdown for action type (key/mouse_click/mouse_move/none)
            action_var = tk.StringVar(value=settings["action"])
            action_box = ttk.Combobox(canvas.inner, textvariable=action_var, values=ACTION_MODES)
            action_box.grid(row=row, column=2)

            # dropdown for key
            key_var = tk.StringVar(value=settings["key"])
            actionType = settings["action"]
            if actionType == "key" or actionType == "mouse_move":
                key_box = ttk.Combobox(canvas.inner, textvariable=key_var, values=SUPPORTED_KEYS)
            else:  # "mouse_click"
                key_box = ttk.Combobox(canvas.inner, textvariable=key_var, values=SUPPORTED_MOUSE_EVENTS)
            key_box.grid(row=row, column=3)

            # dropdown for key mode (tap/hold)
            mode_var = tk.StringVar(value=settings["mode"])
            mode_box = ttk.Combobox(canvas.inner, textvariable=mode_var, values=KEY_MODES)
            mode_box.grid(row=row, column=4)

            # dropdown for retrigger action type
            r_action_var = tk.StringVar(value=settings["retrigger_action"])
            r_action_box = ttk.Combobox(canvas.inner, textvariable=r_action_var, values=ACTION_MODES)
            r_action_box.grid(row=row, column=5)

            # dropdown for retrigger key
            r_key_var = tk.StringVar(value=settings["retrigger_key"])
            actionType = settings["retrigger_action"]
            if actionType == "key" or actionType == "mouse_move":
                r_key_box = ttk.Combobox(canvas.inner, textvariable=r_key_var, values=SUPPORTED_KEYS)
            else:  # "mouse_click"
                r_key_box = ttk.Combobox(canvas.inner, textvariable=r_key_var, values=SUPPORTED_MOUSE_EVENTS)
            r_key_box.grid(row=row, column=6)

            # dropdown for retrigger key mode (tap/hold)
            r_mode_var = tk.StringVar(value=settings["retrigger_mode"])
            r_mode_box = ttk.Combobox(canvas.inner, textvariable=r_mode_var, values=KEY_MODES)
            r_mode_box.grid(row=row, column=7)

            # spinbox for retrigger cooldown
            cd_var = tk.IntVar(value=settings["retrigger_cooldown_ms"])
            cd_box = tk.Spinbox(canvas.inner, from_=0, to=5000, increment=50, textvariable=cd_var)
            cd_box.grid(row=row, column=8)

            # dropdown for enable/disable transition function support
            transition_var = tk.BooleanVar(value=bool(settings["use_finger_transitions"]))
            transition_box = ttk.Checkbutton(canvas.inner, variable=transition_var)
            transition_box.grid(row=row, column=9)

            # dropdown for cursor modes
            c_mode_var = tk.StringVar(value=CURSOR_MODES[settings["cursor_mode"]])
            c_mode_box = ttk.Combobox(canvas.inner, textvariable=c_mode_var, values=CURSOR_MODES)
            c_mode_box.grid(row=row, column=10)

            # slider for cursor sensitivity
            sens_var = tk.DoubleVar(value=settings["cursor_sensitivity"])
            sens_box = tk.Scale(canvas.inner, from_=0.1, to=5.0, resolution=0.1, orient="horizontal", variable=sens_var)
            sens_box.grid(row=row, column=11)

            # Update profile and save
            def save_changes(event=None, hand=hand, gesture_id=gesture_id, enable_var=enable_var,
                             action_var=action_var, key_var=key_var, mode_var=mode_var,
                             r_action_var=r_action_var, r_key_var=r_key_var, r_mode_var=r_mode_var,
                             cd_var=cd_var, transition_var=transition_var, c_mode_var=c_mode_var,
                             sens_var=sens_var):
                profile["gestures"][hand][gesture_id]["enabled"] = int(enable_var.get())
                profile["gestures"][hand][gesture_id]["action"] = action_var.get()
                profile["gestures"][hand][gesture_id]["key"] = key_var.get()
                profile["gestures"][hand][gesture_id]["mode"] = mode_var.get()
                profile["gestures"][hand][gesture_id]["retrigger_action"] = r_action_var.get()
                profile["gestures"][hand][gesture_id]["retrigger_key"] = r_key_var.get()
                profile["gestures"][hand][gesture_id]["retrigger_mode"] = r_mode_var.get()
                profile["gestures"][hand][gesture_id]["retrigger_cooldown_ms"] = int(cd_var.get())
                profile["gestures"][hand][gesture_id]["use_finger_transitions"] = int(transition_var.get())
                profile["gestures"][hand][gesture_id]["cursor_mode"] = CURSOR_MODES.index(c_mode_var.get())
                profile["gestures"][hand][gesture_id]["cursor_sensitivity"] = float(sens_var.get())

                save_profile(profile, path)

            # save when field changes
            enable_box.configure(command=save_changes)
            action_box.bind("<<ComboboxSelected>>", save_changes)
            key_box.bind("<<ComboboxSelected>>", save_changes)
            mode_box.bind("<<ComboboxSelected>>", save_changes)
            r_action_box.bind("<<ComboboxSelected>>", save_changes)
            r_key_box.bind("<<ComboboxSelected>>", save_changes)
            r_mode_box.bind("<<ComboboxSelected>>", save_changes)
            cd_box.configure(command=save_changes)
            cd_box.bind("<<KeyRelease>>", save_changes)
            transition_box.configure(command=save_changes)
            c_mode_box.bind("<<ComboboxSelected>>", save_changes)
            sens_box.configure(command=save_changes)

            row += 1
    
    # transition function settings
    for hand in HANDS:
        settings = profile["transition_functions"][hand]
        tk.Label(canvas.inner, text=f"{hand.upper()} HAND TRANSITIONS", font=("Arial", 14, "bold")).grid(row=row, column=0, sticky="w", pady=10)
        row += 1

        # enable/disable
        tk.Label(canvas.inner, text="Enable").grid(row=row, column=0)
        row += 1

        enable_var = tk.BooleanVar(value=bool(settings["enabled"]))
        enable_box = ttk.Checkbutton(canvas.inner, variable=enable_var)
        enable_box.grid(row=row, column=1)
        row += 1

        tk.Label(canvas.inner, text="Finger").grid(row=row, column=0)
        tk.Label(canvas.inner, text="Extend key").grid(row=row, column=1)
        tk.Label(canvas.inner, text="Retract key").grid(row=row, column=2)
        row += 1

        for i, finger in enumerate(FINGERS):
            tk.Label(canvas.inner, text=finger).grid(row=row, column=0)

            e_key_var = tk.StringVar(value=settings[str(i)]["extend_key"])
            e_key_box = ttk.Combobox(canvas.inner, textvariable=e_key_var, values=SUPPORTED_KEYS)
            e_key_box.grid(row=row, column=1)

            r_key_var = tk.StringVar(value=settings[str(i)]["retract_key"])
            r_key_box = ttk.Combobox(canvas.inner, textvariable=r_key_var, values=SUPPORTED_KEYS)
            r_key_box.grid(row=row, column=2)

            row += 1

            def save_changes(event=None, hand=hand, i=i, e_key_var=e_key_var, r_key_var=r_key_var):
                profile["transition_functions"][hand][str(i)]["extend_key"] = e_key_var.get()
                profile["transition_functions"][hand][str(i)]["retract_key"] = r_key_var.get()
                save_profile(profile, path)
            
            e_key_box.bind("<<ComboboxSelected>>", save_changes)
            r_key_box.bind("<<ComboboxSelected>>", save_changes)
    
    def save_changes(event=None, hand=hand, enable_var=enable_var):
        profile["transition_functions"][hand]["enabled"] = int(enable_var.get())
        save_profile(profile, path)
    
    # save when field changes
    enable_box.configure(command=save_changes)

    # reveal hidden root when closing mapping GUI
    def on_close():
        window.destroy()
        root.deiconify()
    
    window.protocol("WM_DELETE_WINDOW", on_close)

    return window
