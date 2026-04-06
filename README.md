# Gesture Game Controller

This program adapts from a sample program that recognizes hand gestures with a simple MLP using detected Mediapipe key points.
(https://github.com/kinivi/hand-gesture-recognition-mediapipe). Some elements from the original program are employed: debugging functions for drawing landmark skeletons and points onto images, the KeyPointClassifier class, opencv-mediapipe interactions and a modified version of the original 'app.py' file, renamed to 'coordinate_collection.py' as it is used to record mediapipe keypoint coordinates for collecting additional data for gesture classification. Please note that due to file size, the dataset used to train the gesture recognition model is NOT included in this repository.

# Libraries
* OpenCV 4.11.0.86
* mediapipe 0.10.21
* Tensorflow 2.15.0
* scikit-learn 1.7.2
* PyDirectInput 1.0.4
* KerasTuner 1.4.8
* protobuf 4.25.8
* jax 0.4.28
* jaxlib 0.4.28
* ml-dtypes 0.2.0
* pandas 2.3.3 (plotting graphics)
* matplotlib 3.10.0 (plotting graphics)
* seaborn 0.13.2 (confusion matrix)

# Directory
<pre>
│  app.py
│  coordinate_collection.py
│  GestureState.py
│  keypoint_classification.ipynb
│  profile_config.py
│
└─GUIs
|   |  mapping_GUI.py
|   └─ profile_selector_GUI.py
|
├─model
│  └─keypoint_classifier
│     ├─ figures
│     ├─ hyperparameters
|     |     └─ best_hp.json
│     │  keypoint.csv
│     │  keypoint_classifier.hdf5
│     │  keypoint_classifier.py
│     │  keypoint_classifier_full.py
│     │  keypoint_classifier.tflite
│     └─ keypoint_classifier_label.csv
│          
└─mouse_input
|   └─ mouse_wrapper.py
|
└─preprocessing
|   |  extract_landmarks.py
|   |  merge_data.py
|   |  split_sets.py
|   └─ write_data.py
|
└─profiles
│   └─ custom_profiles
│   │     |  default.json
│   │     │  minecraft.json
│   │     └─ void_collector.json
│   └─ prev.txt
│        
└─utils
    └─cvfpscalc.py

</pre>
### app.py
The gesture game controller program.

### coordinate_collection.py
A modified version of the original sample program that is used for collecting mediapipe landmark samples.

### GestureState.py
GestureState class for storing gesture state information

### keypoint_classification.ipynb
This is a model training script for gesture recognition.

### profile_config.py
This program allows for gesture mapping profiles to be edited or created. 

### GUIs
This directory stores scripts for two GUIs related to gesture mapping customisation.
* mapping_GUI.py
* profile_selector_GUI.py

### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Empty file for manual sample collection (keypoint.csv)
* Trained tflite model (keypoint_classifier.tflite)
* Trained full model (keypoint_classifier.hdf5)
* Gesture labels (keypoint_classifier_label.csv)
* Inference module (keypoint_classifier.py)
* Inference module for full model (keypoint_classifier_full.py)
* hyperparameters folder containing best hyperparameters
* figures folder containing model testing results

### mouse_input/mouse_wrapper.py
This file describes the wrapper for Win32's SendInput function for cursor input simulation.

### preprocessing
This directory stores scripts for processing HaGRIDv2 and self-collected landmark data into target dataset files.
* extract_landmarks.py
* merge_data.py
* split_sets.py
* write_data.py

### profiles
This directory stores gesture mapping profiles and prev.txt which lists the most recently used profile.
* custom_profiles: default.json, minecraft.json, void_collector.json
* prev.txt

### utils/cvfpscalc.py
This is a module from the original sample program, used for FPS measurement.
 
# License 
The original sample program is under [Apache v2 license](LICENSE).
