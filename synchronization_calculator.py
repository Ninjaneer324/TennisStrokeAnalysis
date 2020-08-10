import argparse
import logging
import time
import math

import cv2
import numpy as np
import sys
sys.path.insert(1, './tf-pose-estimation/')


from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

centers_vid1 = {}
pairs_vid1 = {}
centers_vid2 = {}
pairs_vid2 = {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video1', type=str, default='')
    parser.add_argument('--video2', type=str, default='')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    
    w, h = model_wh(args.resize)
    e = None
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    #video 1        
    cap = cv2.VideoCapture(args.video1)

    if not cap.isOpened():
        print("Error opening video stream or file")
    frame = 1
    while cap.isOpened():
        ret_val, image = cap.read()
        
        logger.debug('image process+')
        humans = None
        try:
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        except:
            logger.debug('video-complete+')
            break
        if not args.showBG:
            image = np.zeros(image.shape)
        
        logger.debug('postprocess+')
        image, c, pa = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        centers_vid1[frame] = c
        pairs_vid1[frame] = pa
        logger.debug('show+')
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27: #press esc to exit
            break
        frame += 1

    cv2.destroyAllWindows()

    #video 2
    cap = cv2.VideoCapture(args.video2)

    if not cap.isOpened():
        print("Error opening video stream or file")
    frame = 1
    while cap.isOpened():
        ret_val, image = cap.read()
        
        logger.debug('image process+')
        humans = None
        try:
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        except:
            logger.debug('video-complete+')
            break
        if not args.showBG:
            image = np.zeros(image.shape)
        
        logger.debug('postprocess+')
        image, c, pa = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        centers_vid2[frame] = c
        pairs_vid2[frame] = pa
        logger.debug('show+')
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27: #press esc to exit
            break
        frame += 1

    cv2.destroyAllWindows()


    logger.debug('finished+')

    frames = min(len(centers_vid1), len(centers_vid2))
    frame_synched_pairs_count = []
    frame_num_of_matched_pairs_count = []
    for frame in range(1, frames + 1):
        total_similarity = 0
        similar_pairs = list(set(pairs_vid1[frame]).intersection(pairs_vid2[frame]))
        frame_num_of_matched_pairs_count.append(len(similar_pairs))
        for sp in similar_pairs:
            print("Body Parts",sp)
            body_part_1 = sp[0]
            body_part_2 = sp[1]
            v1_point1 = centers_vid1[frame][body_part_1]
            v1_point2 = centers_vid1[frame][body_part_2]
            print(v1_point1)
            print(v1_point2)
            slope1 = (v1_point2[1] - v1_point1[1]) / ((v1_point2[0] - v1_point1[0]) if (v1_point2[0] - v1_point1[0]) != 0 else .00000001)

            v2_point1 = centers_vid2[frame][body_part_1]
            v2_point2 = centers_vid2[frame][body_part_2]
            print(v2_point1)
            print(v2_point2)
            slope2 = (v2_point2[1] - v2_point1[1]) / ((v2_point2[0] - v2_point1[0]) if (v2_point2[0] - v2_point1[0]) != 0 else .00000001)

            deviation = abs(slope1 - slope2) / (slope1 if slope1 != 0 else .00000001)
            if deviation <= .20:
                total_similarity += 1
        frame_synched_pairs_count.append(total_similarity)
    
    print("Synchronization Score:", np.mean([frame_synched_pairs_count[i]/frame_num_of_matched_pairs_count[i] for i in range(len(frame_synched_pairs_count))]) * 100)



    
    