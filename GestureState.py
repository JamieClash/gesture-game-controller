class GestureState:
    def __init__(self, rel_gain=25, pan_speed=50, angle_gain=1600):
        # rel_gain=25, pan_speed=20, angle_gain=400 (I think can be higher) <-- minecraft
        # rel_gain=25, pan_speed=20 (I think lower), angle_gain=400 (I think lower) < -- void collector

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

        # movement threshold for gestures that control the mouse cursor.
        self.cursor_movement_threshold = 0.01
        
        # smoothing coefficient for cursor movements
        self.cursor_alpha = 0.2

        # cursor movement deltas for smoothing
        self.prev_dxs = [0, 0]
        self.prev_dys = [0, 0]

        # 'starting' index finger tip position (8) stored as the initial position for relative cursor movement. 
        # cursor mode 1
        self.prev_rel_pos = [None, None]
        self.rel_gain = rel_gain

        # 'starting' wrist position (0) stored as an 'origin' for panning cursor movement.
        # cursor mode 2
        self.base_panning_origins = [None, None]
        self.panning_threshold = 10  # 3
        self.panning_max_speed = pan_speed
        self.panning_gain = 0.3

        # angles used for angle-based cursor movement. Angle between wrist (0) and second joint of middle finger (11)
        # cursor mode 3
        self.prev_angle_vectors = [None, None]
        self.angle_threshold = 2  # in degrees
        self.angle_gain = angle_gain
    
    def reset_deltas(self, index):
        self.prev_dxs[index] = 0
        self.prev_dys[index] = 0
        