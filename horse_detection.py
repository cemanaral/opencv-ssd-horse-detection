import imageio
import torch
import cv2

from data import *
from ssd import *

def detect(frame, net, transform):
    height, width, _ = frame.shape
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = x.unsqueeze(0)
    with torch.no_grad():
        y = net(x)
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, VOC_CLASSES[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame


net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))
transform = BaseTransform(net.size, (104/256, 117/256, 123/256))
reader = imageio.get_reader('epic_horses.mp4')
fps = reader.get_meta_data()['fps']
nframes = reader.get_meta_data()['nframes']
writer = imageio.get_writer('output.mp4', fps=fps,  codec='mpeg4')

i = 1
for frame in reader:
    print("Processing frame: %d / %d" % (i, nframes))
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    i += 1
    
writer.close()
