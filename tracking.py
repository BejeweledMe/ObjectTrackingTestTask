import warnings
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from models import *
from sort import *
from utils import utils
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--video-path', type=str, default='campus4-c0.avi')
arg('--write-path', type=str, default='det-campus4-c0.avi')
arg('--config-path', type=str, default='config/yolov3.cfg')
arg('--weights-path', type=str, default='config/yolov3.weights')
arg('--class-path', type=str, default='config/coco.names')
arg('--img-size', type=int, default=416)
arg('--conf-thres', type=float, default=0.8)
arg('--nms-thres', type=float, default=0.4)
args = parser.parse_args()

videopath = args.video_path
writepath = args.write_path
config_path = args.config_path
weights_path = args.weights_path
class_path = args.class_path
img_size = args.img_size
conf_thres = args.conf_thres
nms_thres = args.nms_thres

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.to(device)
model.eval()
classes = utils.load_classes(class_path)


def detect_image(img):
    '''
    This function applies transforms to image [resize, padding, to tensor],
    and then applies detection algorithm

    :param img: PIL image
    :return: detections
    '''
    # scale and pad image
    ratio = min(img_size / img.size[0], img_size / img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                         transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0),
                                                         max(int((imh - imw) / 2), 0),
                                                         max(int((imw - imh) / 2), 0)),
                                                        (128, 128, 128)),
                                         transforms.ToTensor(),
                                         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0).to(device)
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(image_tensor)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


def main():
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # initialize video capture
    cap = cv2.VideoCapture(videopath)
    # get info about fps and resolution
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # initialize video writer
    codec = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(writepath, codec, fps, res)
    mot_tracker = Sort()

    frames = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # convert to PIL image and detect
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        # check for detected objects
        if detections is not None:
            # update tracker info
            tracked_objects = mot_tracker.update(detections.cpu())

            # iterate for tracked objects
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                cls = classes[int(cls_pred)]
                # update frame with bbox info
                cv2.rectangle(frame, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
                cv2.rectangle(frame, (x1, y1 - 35), (x1 + len(cls) * 19 + 60, y1), color, -1)
                cv2.putText(frame,
                            cls + "-" + str(int(obj_id)),
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 3)

        cv2.imshow('stream', frame)

        frames += 1
        # write frame
        out.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # close opencv stream processes
    cap.release()
    out.release()

    print(f'Total frames: {frames}')


if __name__ == '__main__':
    main()
