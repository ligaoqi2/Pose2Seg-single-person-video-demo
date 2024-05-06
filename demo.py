import argparse

import cv2
import numpy as np
from tqdm import tqdm

from modeling.build_model import Pose2Seg
from datasets.CocoDatasetInfo import CocoDatasetInfo, annToMask
from pycocotools import mask as maskUtils

from matplotlib import pyplot as plt

from openpose.openpose import Openpose


def draw_mask(img, mask, color, transparency):

    mask_rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
    color_with_alpha = color + (int(255 * transparency),)
    mask_colored = np.zeros_like(mask_rgba)
    mask_colored[mask != 0] = color_with_alpha

    result = cv2.cvtColor(cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA), 1, mask_colored, 0.5, 0), cv2.COLOR_RGBA2BGR)

    return result


def inference(pose_model, seg_model, video_path):
    seg_model.eval()

    cap = cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1]
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = './output/' + 'Pose2Seg_' + video_name
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    color = (255, 0, 0)  # Green color
    transparency = 0.5   # 50% transparency

    for i in tqdm(range(count)):
        ret, img = cap.read()
        if ret:
            pose_output, scores, poses_with_scores = pose_model.detect(img, precise=False)

            # single person
            if pose_output.shape[0] == 1:
                pass
            else:
                highest_person_result = []
                for j in range(pose_output.shape[0]):
                    if j == 0:
                        highest_person_result.append(pose_output[j])
                    else:
                        if pose_output[j][13][1] > highest_person_result[0][13][1]:
                            highest_person_result[0] = pose_output[j]

                pose_output = np.array(highest_person_result).reshape([1, 18, 4])

            seg_input_kpts = np.delete(pose_output, 1, axis=1)[:, :, :3]
            seg_output = seg_model([img], [seg_input_kpts], None)

            result = draw_mask(img.copy(), seg_output[0][0], color, transparency)

            # multi person will be completed in the future
            # for mask in seg_output[0]:
            #     print("ok")

            out.write(result)

    print('video has been saved as {}'.format(save_path))


if __name__ == '__main__':
    video_path = "./your-own-video-path"

    pose_model = Openpose(weights_file="./openpose/models/posenet.pth", training=False)

    seg_model = Pose2Seg().cuda()
    seg_model.init("./pose2seg_release.pkl")

    inference(pose_model=pose_model, seg_model=seg_model, video_path=video_path)
