# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import cv2
import torch

import mocap_utils.demo_utils as demo_utils

from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector

import time
import argparse

class DemoOptions():

    def __init__(self):
        parser = argparse.ArgumentParser()

        # parser.add_argument('--checkpoint', required=False, default=default_checkpoint, help='Path to pretrained checkpoint')
        default_checkpoint_body_smpl ='./extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt'
        parser.add_argument('--checkpoint_body_smpl', required=False, default=default_checkpoint_body_smpl, help='Path to pretrained checkpoint')
        default_checkpoint_body_smplx ='./extra_data/body_module/pretrained_weights/smplx-03-28-46060-w_spin_mlc3d_46582-2089_2020_03_28-21_56_16.pt'
        parser.add_argument('--checkpoint_body_smplx', required=False, default=default_checkpoint_body_smplx, help='Path to pretrained checkpoint')
        default_checkpoint_hand = "./extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
        parser.add_argument('--checkpoint_hand', required=False, default=default_checkpoint_hand, help='Path to pretrained checkpoint')

        # Other options
        parser.add_argument('--smpl_dir', type=str, default='./extra_data/smpl/', help='Folder where smpl files are located.')

        # Hand mocap specific options
        parser.add_argument('--view_type', type=str, default='third_view', choices=['third_view', 'ego_centric'],
            help = "The view type of input. It could be ego-centric (such as epic kitchen) or third view")
        parser.add_argument('--crop_type', type=str, default='no_crop', choices=['hand_crop', 'no_crop'],
            help = """ 'hand_crop' means the hand are central cropped in input. (left hand should be flipped to right).
                        'no_crop' means hand detection is required to obtain hand bbox""")

        # renderer
        parser.add_argument("--renderer_type", type=str, default="opengl",
            choices=['pytorch3d', 'opendr', 'opengl_gui', 'opengl'], help="type of renderer to use")

        self.parser = parser


    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt



def run_hand_mocap(args, bbox_detector, hand_mocap, visualizer):
    #Set up input data (images or webcam)
    input_type, input_data = demo_utils.setup_input(args)
    cur_frame = 0
    video_frame = 0

    while True:
        # load data
        start_time = time.time()

        _, img_original_bgr = input_data.read()
        if video_frame < cur_frame:
            video_frame += 1
            continue

        cur_frame +=1
        if img_original_bgr is None:
            break   

        # bbox detection
        if args.crop_type == 'hand_crop':
            # hand already cropped, thererore, no need for detection
            img_h, img_w = img_original_bgr.shape[:2]
            hand_bbox_list = [ dict(right_hand = np.array([0, 0, img_w, img_h])) ]
        else:            
            # Input images has other body part or hand not cropped.
            # Use hand detection model & body detector for hand detection
            assert args.crop_type == 'no_crop'
            detect_output = bbox_detector.detect_hand_bbox(img_original_bgr.copy())
            print("detect_output", detect_output)
            _, body_bbox_list, hand_bbox_list, _ = detect_output
        
        # save the obtained body & hand bbox to json file

        if len(hand_bbox_list) < 1:
            # print(f"No hand deteced: {image_path}")
            continue
    
        # Hand Pose Regression
        pred_output_list = hand_mocap.regress(
                img_original_bgr, hand_bbox_list, add_margin=True)
        assert len(hand_bbox_list) == len(body_bbox_list)
        assert len(body_bbox_list) == len(pred_output_list)

        # print("pred_output_list", pred_output_list)

        # extract mesh for rendering (vertices in image space and faces) from pred_output_list
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

        # visualize
        res_img = visualizer.visualize(
            img_original_bgr,
            pred_mesh_list = pred_mesh_list,
            hand_bbox_list = hand_bbox_list)

        time_elapsed = time.time() - start_time
        print(f"FPS: {1/time_elapsed:.2f}")

    # When everything done, release the capture
    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()

  
def main():
    args = DemoOptions().parse()
    args.use_smplx = True
    args.input_path = "webcam"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    #Set Bbox detector
    bbox_detector =  HandBboxDetector(args.view_type, device)

    # Set Mocap regressor
    hand_mocap = HandMocap(args.checkpoint_hand, args.smpl_dir, device = device)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)

    # run
    run_hand_mocap(args, bbox_detector, hand_mocap, visualizer)
   

if __name__ == '__main__':
    main()
