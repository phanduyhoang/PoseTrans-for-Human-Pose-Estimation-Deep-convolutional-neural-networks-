#affine transformation: rotation, scaling 

import numpy as np
import cv2

class PoseTransformation:
    def __init__(self, scale_range=(0.75,2.15), rot_range=(-35,35)):
        self.scale_range=scale_range
        self.rot_range=rot_range
    def apply_transformation(self,image, keypoints, limb_indices):

        src_pt,dst_pt=keypoints[limb_indices[0]], keypoints[limb_indices[1]]
        center=((src_pt[0]+dst_pt[0])//2, (src_pt[1]+dst_pt[1])//2)

        scale=np.random.uniform(*self.scale_range)
        angle=np.random.uniform(*self.rot_range)


        #affine transformation
        M=cv2.getRotationMatrix2D(center, angle, scale)
        transformed_image=cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Transform keypoints
        ones=np.ones((keypoints.shape[0],1))
        keypoints_homogeneous=np.hstack([keypoints[:,:2],ones])
        transformed_keypoints=np.dot(M, keypoints_homogeneous.T).T
        return transformed_image, transformed_keypoints