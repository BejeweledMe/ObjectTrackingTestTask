import os


# common values
config_path = 'config/yolov3.cfg'
weights_path = 'config/yolov3.weights'
class_path = 'config/coco.names'
# use values from docs for better results
img_size = 416
conf_thres = 0.8
nms_thres = 0.4

# first video
videopath = 'data/campus4-c0.avi'
writepath = 'data/det-campus4-c0.avi'
os.system(f'python -m tracking --config-path {config_path} --weights-path {weights_path} \
--class-path {class_path} --img-size {img_size} --conf-thres {conf_thres} --nms-thres {nms_thres} \
--video-path {videopath} --write-path {writepath}')

# second video
videopath = 'data/campus4-c1.avi'
writepath = 'data/det-campus4-c1.avi'
os.system(f'python -m tracking --config-path {config_path} --weights-path {weights_path} \
--class-path {class_path} --img-size {img_size} --conf-thres {conf_thres} --nms-thres {nms_thres} \
 --video-path {videopath} --write-path {writepath}')

# third video
videopath = 'data/campus4-c2.avi'
writepath = 'data/det-campus4-c2.avi'
os.system(f'python -m tracking --config-path {config_path} --weights-path {weights_path} \
--class-path {class_path} --img-size {img_size} --conf-thres {conf_thres} --nms-thres {nms_thres} \
 --video-path {videopath} --write-path {writepath}')
