# PyTorch Object Detection and Tracking
Object detection in images, and tracking across video frames

<img src="https://github.com/BejeweledMe/ObjectTrackingTestTask/blob/main/tracking.gif?raw=true" width="360" height="288" />

This is my testtask for eora.ru on the middle cv engineer position

### Algorithm
1. Take a yolo_v3 model
2. Download pretrained on COCO dataset weights with `config/download_weights.sh` and add it to `config` folder
3. Download seq. 1 cam. 0-2 videos from https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/ 
and add it to `data` folder
4. Load weights in model
5. Open video with `cv2.VideoCapture`
6. Iterate over stream and detect objects on frames
7. Use tracker from `sort.py` on detected objects
8. Draw bounding boxes and ids for objects on frames and write it with `cv2.VideoWriter`
9. Close all opencv processes

### Run
To run predict on videos, you can use `bash run_tracking.sh` in terminal to run `run_tracking.py` file, which will run `tracking.py` for 3 videos

Or you can run `tracking.py` from terminal with arguments

###### Argparse arguments
`--video-path` - path to read video

`--write-path` - path to write video

`--config-path` - path to model config

`--weights-path'` - path to model weights

`--class-path` - path to file with class names

`--img-size` - image size for model

`--conf-thres` - confidence threshold

`--nms-thres` - threshold for NMS



# Links

Thanks to https://github.com/cfotache/pytorch_objectdetecttrack for the code to pretrained model