# Gesture Game Controller

This program adapts from a sample program that recognizes hand gestures with a simple MLP using detected Mediapipe key points.
(https://github.com/kinivi/hand-gesture-recognition-mediapipe). Some elements from the original program are employed: debugging functions for drawing landmark skeletons and points onto images, the KeyPointClassifier classes (the latter is kept), opencv-mediapipe interactions and a modified version of the original 'app.py' file, renamed to 'coordinate_collection.py' as it is used to record mediapipe keypoint coordinates for collecting additional data for gesture classification.

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
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py

</pre>
### app.py
This is a sample program for inference.<br>
In addition, learning data (key points) for hand sign recognition,<br>
You can also collect training data (index finger coordinate history) for finger gesture recognition.

### keypoint_classification.ipynb
This is a model training script for hand sign recognition.

### point_history_classification.ipynb
This is a model training script for finger gesture recognition.

### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### utils/cvfpscalc.py
This is a module for FPS measurement.
 
# License 
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).
