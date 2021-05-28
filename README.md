# PyTorch Object Detection and Tracking
Object detection in images, and tracking across video frames

Thanks to https://github.com/cfotache/pytorch_objectdetecttrack

<img src="https://github.com/BejeweledMe/ObjectTrackingTestTask/blob/main/tracking.gif?raw=true" width="360" height="288" />

### Short description
This is my testtask for eora.ru for the middle cv engineer position

The task was to run detection and tracking algorithms on video from 
https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/

For this task I have used pytorch and opencv libraries, 
the model was yolo_v3 pretrained on the COCO dataset

### Argparse arguments
`--video-path, type=str, default='campus4-c0.avi'`

`--write-path, type=str, default='det-campus4-c0.avi'`

`--config-path, type=str, default='config/yolov3.cfg'`

`--weights-path, type=str, default='config/yolov3.weights'`

`--class-path, type=str, default='config/coco.names'`

`--img-size, type=int, default=416`

`--conf-thres, type=float, default=0.8`

`--nms-thres, type=float, default=0.4`

### Run
Download seq. 1 cam. 0-2 videos from https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/ 
and add it to `data` folder

Download weights with `config/download_weights.sh` and add it to `config` folder

To run predict on videos, you can run `run_tracking.py` file
