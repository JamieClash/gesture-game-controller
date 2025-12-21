#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import json
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import pydirectinput

import os
import subprocess
import time
import threading

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

from GestureState import GestureState

MAX_NUM_HANDS = 2

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

DEBUG = False  # if True, shows bounding boxes, landmarks and gesture classification (causes lag)

profile_folder = "profiles/custom_profiles/"
default_profile_path = profile_folder + "default.json"

# used to retrieve previously used gesture-key mapping profile
prev_profile_path = "profiles/prev.txt"

# used for selecting profiles
profile_selector_path = "GUIs/profile_selector_GUI.py"

# used to keep track of whether profile selection UI is open
current_process = None 

# used for querying the profile dictionary when having indices instead of strings
hand_labels = ["left", "right"]
bad_keys = ["NULL", "left_click", "right_click", "middle_click", "double_click", "triple_click"]

# paths for gesture labels and keypoint path (latter is disabled)
keypoint_label_path = "model/keypoint_classifier/keypoint_classifier_label.csv"
keypoint_path = "model/keypoint_classifier/keypoint.csv"

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=MIN_DETECTION_CONFIDENCE)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=MIN_TRACKING_CONFIDENCE)

    args = parser.parse_args()

    return args

def get_prev_path():
    with open(prev_profile_path, "r") as f:
        path = f.readline()
    return path

def set_prev_path(profile_path):
    with open(prev_profile_path, "w") as f:
        f.write(profile_path)

def get_mappingDict(profile_path):
    with open(profile_path, "r") as f:
        data = json.load(f)
    return data

def open_profile_selector():
    global current_process

    # selector already open.
    if current_process is not None and current_process.poll() is None:
        return

    current_process = subprocess.Popen(["python", profile_selector_path], stdout=subprocess.PIPE, text=True)

def check_selection():
    global current_process

    if current_process is None:
        return None

    if current_process.poll() is not None:
        path = current_process.stdout.read().strip()
        current_process = None
        if os.path.isfile(path):
            return path
    
    return None

# variables for threads
camera_frame = None
camera_lock = threading.Lock()

perception_output = None
perception_lock = threading.Lock()

render_snapshot = None
render_lock = threading.Lock()

cv_output = None
cv_lock = threading.Lock()

stop_event = threading.Event()

# variables for gesture-action mapping profiles
profile = None
profile_name = None

# capture device variable
cap = None

def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # set up gesture mapping
    prev_path = get_prev_path()

    global profile
    global profile_name

    profile = get_mappingDict(prev_path)
    profile_name = profile["profile_name"]

    # Camera preparation ###############################################################
    global cap
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels ###########################################################
    with open(keypoint_label_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    gesture_state = GestureState()

    # functions for the game controller
    def apply_action(curr_gesture, retrigger=False):
        if curr_gesture is None:
            return

        # not applicable if gesture supports transition function
        if curr_gesture["use_finger_transitions"] or not curr_gesture["enabled"]:
            return
        
        def apply_click(click):
            if click == "left_click":
                pydirectinput.leftClick()
            elif click == "right_click":
                pydirectinput.rightClick()
            elif click == "middle_click":
                pydirectinput.middleClick()
            elif click == "double_click":
                pydirectinput.doubleClick()
            elif click == "triple_click":
                pydirectinput.tripleClick()
        
        # apply action if applicable
        if retrigger:
            if curr_gesture["retrigger_action"] == "mouse_click":
                apply_click(curr_gesture["retrigger_key"])
            elif curr_gesture["retrigger_key"] not in bad_keys:
                if curr_gesture["retrigger_mode"] == "hold":
                    pydirectinput.keyDown(curr_gesture["retrigger_key"])
                else:
                    pydirectinput.press(curr_gesture["retrigger_key"])
        else:
            if curr_gesture["action"] == "mouse_click":
                apply_click(curr_gesture["key"])
            elif curr_gesture["key"] not in bad_keys:
                if curr_gesture["mode"] == "hold":
                    pydirectinput.keyDown(curr_gesture["key"])
                else:
                    pydirectinput.press(curr_gesture["key"])

    def end_action(prev_gesture, retrigger=False):
        if prev_gesture is None:
            return

        # not applicable if gesture supports transition function
        if prev_gesture["use_finger_transitions"] or not prev_gesture["enabled"]:
            return
        
        # release key if applicable
        if (retrigger and prev_gesture["retrigger_action"] != "mouse_click" and prev_gesture["retrigger_mode"] == "hold" 
            and prev_gesture["retrigger_key"] not in bad_keys):
            pydirectinput.keyUp(prev_gesture["retrigger_key"])
        elif (not retrigger and prev_gesture["action"] != "mouse_click" and prev_gesture["mode"] == "hold" 
            and prev_gesture["key"] not in bad_keys):
            pydirectinput.keyUp(prev_gesture["key"])
    
    def apply_transition_function(handedness, prev_gesture, curr_gesture):
        global profile

        def press_key(key):
            if key != "NULL":
                pydirectinput.press(key)
        
        def supported(gesture):
            return (gesture["use_finger_transitions"] and gesture["enabled"])

        transition_profile = profile["transition_functions"][hand_labels[handedness]]
        extend_keys = [transition_profile[str(i)]["extend_key"] for i in range(5)]
        retract_keys = [transition_profile[str(i)]["retract_key"] for i in range(5)]

        if not transition_profile["enabled"]:
            return
        
        if prev_gesture is None and curr_gesture is None:
            return
        elif prev_gesture is None and curr_gesture is not None:
            # treat prev_gesture as [0,0,0,0,0]
            if supported(curr_gesture):
                curr_state = curr_gesture["hand_state"]
                for i, s in enumerate(curr_state):
                    if s:
                        press_key(extend_keys[i])
            return
        elif prev_gesture is not None and curr_gesture is None:
            # treat curr_gesture as [0,0,0,0,0]
            if supported(prev_gesture):
                prev_state = prev_gesture["hand_state"]
                for i, s in enumerate(prev_state):
                    if s:
                        press_key(retract_keys[i])
            return

        prev_support = supported(prev_gesture)
        curr_support = supported(curr_gesture)

        if (not prev_support) and (not curr_support):
            return

        prev_state = prev_gesture["hand_state"]
        curr_state = curr_gesture["hand_state"]

        if prev_support and curr_support:
            # transform from prev_state to curr_state with transition keys
            for i in range(len(prev_state)):
                p_s = prev_state[i]
                c_s = curr_state[i]

                if p_s > c_s:
                    press_key(retract_keys[i])
                elif c_s > p_s:
                    press_key(extend_keys[i])

        elif prev_support:
            # reset prev_state to [0,0,0,0,0] with transition keys
            for i, s in enumerate(prev_state):
                if s:
                    press_key(retract_keys[i])
        elif curr_support:
            # set [0,0,0,0,0] to curr_state with transition keys
            for i, s in enumerate(curr_state):
                if s:
                    press_key(extend_keys[i])

    ### THREADS ###
    # thread for camera frames
    def camera_loop():
        global camera_frame
        global cap

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                stop_event.set()
            with camera_lock:
                camera_frame = cv.flip(frame, 1)
    
    # thread for running mediapipe and classification
    def perception_loop():
        global perception_output

        while not stop_event.is_set():
            with camera_lock:
                frame = None if camera_frame is None else camera_frame.copy()
            
            if frame is None:
                continue
            
            debug_image = copy.deepcopy(frame)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # run mediapipe
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True

            # list variables to accomodate both handed gestures
            gesture_ids = [-1, -1]
            brects = [[], []]
            landmark_lists = [[], []]

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    h_index = 0 if handedness.classification[0].label[0:] == "Left" else 1

                    # bounding box and landmark calculations
                    brects[h_index] = calc_bounding_rect(debug_image, hand_landmarks)

                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    landmark_lists[h_index] = landmark_list

                    # convert to relative + normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                    # gesture classification
                    gesture_ids[h_index] = keypoint_classifier(pre_processed_landmark_list)

            with perception_lock:
                perception_output = {
                    "gesture_ids": gesture_ids,
                    "landmarks": landmark_lists,
                    "brects": brects,
                    "debug_image": debug_image
                }

    def gesture_logic_loop():
        global perception_output
        global render_snapshot
        global profile

        while not stop_event.is_set():
            with perception_lock:
                data = None if perception_output is None else copy.deepcopy(perception_output)

            if data is None:
                continue

            # update gesture state, applying/ending actions if necessary
            for h_index in range(2):
                gesture_id = data["gesture_ids"][h_index]

                # end actions for hands that are no longer in camera view
                if gesture_id == -1:
                    if gesture_state.prev_gestures[h_index] is not None:
                        pf = profile["gestures"][hand_labels[h_index]][str(gesture_state.prev_gestures[h_index])]
                        end_action(pf)
                        end_action(pf, True)
                        apply_transition_function(h_index, hand_profile[str(gesture_state.prev_gestures[h_index])], None)
                        gesture_state.prev_gestures[h_index] = None
                    continue

                landmark_list = data["landmarks"][h_index]
                brect = data["brects"][h_index]

                hand_profile = profile["gestures"][hand_labels[h_index]]
                curr_gesture = hand_profile[str(gesture_id)]
                
                # check if gesture for this hand has changed.
                prev_id = gesture_state.prev_gestures[h_index]
                if prev_id != gesture_id:
                    prev_gesture = None if (prev_id is None) else (
                        hand_profile[str(gesture_state.prev_gestures[h_index])])

                    # apply transition function if applicable
                    apply_transition_function(h_index, prev_gesture, curr_gesture)
                    
                    # end previous actions if applicable
                    end_action(prev_gesture)
                    end_action(prev_gesture, True)

                    # update everything stored according to gesture
                    cursor_mode = curr_gesture["cursor_mode"]

                    if curr_gesture["action"] == "mouse_move" and cursor_mode != 0:
                        if cursor_mode == 1:
                            gesture_state.base_absolute_origins[h_index] = landmark_list[8]
                        elif cursor_mode == 2:
                            gesture_state.base_panning_origins[h_index] = landmark_list[0]
                        elif cursor_mode == 3:
                            curr_angle = get_angle(landmark_list[0], landmark_list[11])
                            if gesture_state.prev_angles[h_index] is not None:
                                delta_angle = curr_angle - gesture_state.prev_angles[h_index]
                            else:
                                delta_angle = curr_angle
                            # do some stuff with angles idk (actually isn't this for the section below)

                    # apply current gesture's action if applicable
                    apply_action(curr_gesture)
                    
                    gesture_state.prev_gestures[h_index] = gesture_id

                    gesture_state.base_retrigger_area[h_index] = calc_area(brect)
                    gesture_state.last_retrigger_time[h_index] = 0
                    gesture_state.retrigger_cooldown[h_index] = curr_gesture["retrigger_cooldown_ms"]
                    gesture_state.retrigger_ready[h_index] = True
                else:
                    cursor_mode = curr_gesture["cursor_mode"]
                    if curr_gesture["action"] == "mouse_move" and cursor_mode != 0:
                        sens = curr_gesture["cursor_sensitivity"]

                        # do corresponding mouse movement stuff (taking into account sensitivity)
                        # ...

                    # check for retriggers here, taking into account retrigger_cooldown_ms and
                    # whether we need to end action for held retrigger actions (or perform them).
                    curr_area = calc_area(brect)
                    t = time.time() * 1000  # ms

                    # retrigger gesture
                    if (gesture_state.retrigger_ready[h_index] and 
                        (curr_area / gesture_state.base_retrigger_area[h_index] > gesture_state.retrigger_threshold) and 
                        (t - gesture_state.last_retrigger_time[h_index] > gesture_state.retrigger_cooldown[h_index])):
                        gesture_state.last_retrigger_time[h_index] = t
                        gesture_state.retrigger_ready[h_index] = False
                        apply_action(curr_gesture, True)
                    # stop retriggered gesture
                    elif ((not gesture_state.retrigger_ready[h_index]) and 
                    (curr_area / gesture_state.base_retrigger_area[h_index] < gesture_state.relax_threshold)):
                        gesture_state.retrigger_ready[h_index] = True
                        end_action(curr_gesture, True)
            
            # provide data for rendering debug image if needed.
            if DEBUG:
                with render_lock:
                    render_snapshot = data.copy()
    
    def render_loop():
        global render_snapshot
        global cv_output

        while not stop_event.is_set():
            with render_lock:
                data = None if render_snapshot is None else render_snapshot.copy()
            
            if data is None:
                continue
            
            debug_image = data["debug_image"]
            landmarks = data["landmarks"]
            brects = data["brects"]
            gesture_ids = data["gesture_ids"]

            # draw onto debug image
            for h_index in range(2):
                gesture_id = gesture_ids[h_index]

                if gesture_id == -1:
                    continue
            
                brect = brects[h_index]
                landmark_list = landmarks[h_index]

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)

                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    h_index,
                    keypoint_classifier_labels[gesture_id]
                )

            with cv_lock:
                cv_output = debug_image

    # setup and run threads
    threads = [threading.Thread(target=camera_loop), threading.Thread(target=perception_loop), 
               threading.Thread(target=gesture_logic_loop)]

    # render debug information onto camera frames if debugging
    if DEBUG:
        threads.append(threading.Thread(target=render_loop))
    
    for t in threads:
        t.start()
    
    while not stop_event.is_set():
        # draw debug image
        if DEBUG:
            with cv_lock:
                debug_image = cv_output.copy() if cv_output is not None else None
                
        # draw normal image without bounding boxes or gesture name
        else:
            with camera_lock:
                debug_image = camera_frame.copy() if camera_frame is not None else None

        fps = cvFpsCalc.get()

        key = cv.waitKey(10)
        if key == 27:  # ESC
            stop_event.set()
        elif key == 101:  # e
            open_profile_selector()

        # change profile if it has been modified
        profile_path = check_selection()
        if profile_path:
            set_prev_path(profile_path)
            profile = get_mappingDict(profile_path)
            profile_name = profile["profile_name"]

        if debug_image is not None:
            debug_image = draw_info(debug_image, fps, profile_name)
            cv.imshow("Gesture Game Controller", debug_image)

    for t in threads:
        t.join()

    cap.release()
    cv.destroyAllWindows()

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_area(coord_list):
    x0 = coord_list[0]
    y0 = coord_list[1]
    x1 = coord_list[2]
    y1 = coord_list[3]

    return (x1 - x0) * (y1 - y0)


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def get_angle(wrist_coords, mid_coords):
    return 0

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    norms = []

    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
        norms.append(np.linalg.norm([temp_landmark_list[index][0], temp_landmark_list[index][1]]))

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_norm = max(norms) if norms else 1e-6

    def normalize_(n):
        return n / max_norm

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness_index, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = hand_labels[handedness_index].title()

    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

def draw_info(image, fps, profile_name=None):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    cv.putText(image, "Current profile: " + profile_name, (10, 90),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv.LINE_AA)
    return image

if __name__ == '__main__':
    main()
