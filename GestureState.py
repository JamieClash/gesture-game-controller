class GestureState:
    def __init__(self):
        # [left_gesture_index, right_gesture_index] --> Access gesture settings via profile["gestures"]["left" || "right"][gesture_index]
        # used for transition functions and releasing held keys
        self.prev_gestures = [None, None]

        # baseline gesture bounding box area. If increased in size beyond threshold, 'retrigger' a gesture's mapped action.
        self.base_retrigger_area = [None, None]
        self.last_retrigger_time = [0, 0]
        self.retrigger_cooldown = [0, 0]
        self.retrigger_ready = [True, True]

        self.retrigger_threshold = 1.25
        self.relax_threshold = 1.1  # gesture must return to this size before re-triggering again

        # 'starting' index finger tip position (8) stored as an 'origin' for absolute cursor movement. 
        # cursor mode 1
        self.base_absolute_origins = [None, None]

        # 'starting' wrist position (0) stored as an 'origin' for panning cursor movement.
        # cursor mode 2
        self.base_panning_origins = [None, None]
        self.panning_threshold = 0.2  # threshold for triggering cursor panning

        # angles used for angle-based cursor movement. Angle between wrist (0) and second joint of middle finger (11)
        # cursor mode 3
        self.prev_angles = [None, None]
        