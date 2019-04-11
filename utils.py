from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
import argparse
import math
import scipy.io as sio


# Object Annotation
class Object3d(object):
    """
    3D Object Label
    """

    def __init__(self, label_file_line):
        """
        :param label_file_line: each line in <label>.txt
        """
        super(Object3d, self).__init__()
        data = label_file_line.split(' ')
        data = [float(x) for x in data]

        self.tracking_id = int(data[0])  # tracking id
        self.task_id = int(data[1])  # task id
        self.behavior_id = int(data[2])  # behavior id
        self.class_id = int(data[3])  # class id
        self.action_id = int(data[4])  # action id
        """
        extract 3D bounding box information
        """
        self.t = [
            data[5],
            data[6],
            (data[12] +
             data[11]) /
            2]  # coordinate x,y,z
        self.l = data[7]  # length
        self.w = data[8]  # width
        self.h = data[12] - data[11]  # height
        '''
        TODO: This Representation is wrong
        self.ry = math.atan(
            data[9] / (data[10] + 1e-16))  # yaw angle (around Y-axis in LiDAR coordinates) [-pi..pi]
        '''
        self.sin_ry = data[9]     # sin(yaw angle)
        self.cos_ry = data[10]    # cos(yaw angle)

    def print_object(self):
        print('3d bbox h,w,l: %f, %f, %f' %
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), (%f,%f)' %
              (self.t[0], self.t[1], self.t[2], self.sin_ry, self.cos_ry))

# Calibration


class Calibration(object):
    """
    Calibration matrices and utils
    3D XYZ in Object Annotation are in velodyne coordinate
    Points in <lidar>.bin are in Velodyne coordinate
    Image pixels in <image>.jpg are in Image coordinate
    """

    def __init__(self, Tr_velo_to_cam_path, cam_intrinsics_file_path):
        """
        :param Tr_velo_to_cam_path: transform from velodyne to cam coordinate frame
        :param cam_intrinsics_file_path: Camera matrix
        """

        calib_matrix = sio.loadmat(Tr_velo_to_cam_path)
        rotation_matrix = calib_matrix["R"]
        translation_matrix = calib_matrix["t"]
        Tr_velo_to_cam = np.hstack((rotation_matrix, translation_matrix))
        Tr_velo_to_cam[:, 1] = Tr_velo_to_cam[:, 1] * (-1)

        """
        3x4    Tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        """
        self.Tr_velo_to_cam = Tr_velo_to_cam  # V2C

        """
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.

        """
        self.ro_rect = np.eye(3)

        cam_intrinsics_matrix = sio.loadmat(cam_intrinsics_file_path)
        self.cam_fc1, self.cam_fc2 = cam_intrinsics_matrix["fc"]
        self.cam_cc = cam_intrinsics_matrix["cc"]

        """
        3x4    cam_matrix_p    Camera  cam_matrix_p. Contains extrinsic
                                        and intrinsic parameters.

        """

        self.cam_matrix_p = np.array([[self.cam_fc1, 0, self.cam_cc[0], 0],
                                      [0, self.cam_fc2, self.cam_cc[1], 0],
                                      [0, 0, 1, 0]], dtype=np.float32)

        # print(self.cam_matrix_p)


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def load_image(img_filename):
    img = cv2.imread(img_filename)
    img = img[..., ::-1]
    return img


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float64)
    scan = scan.reshape((-1, 4))
    return scan


if __name__ == "__main__":
    """
    label_filename = "/home/doujian/Desktop/Dataset/label/000001.txt"
    objects = read_label(label_filename)
    for object in objects:
        object.print_object()
    """

    calib = Calibration(
        "/home/doujian/Desktop/Rt.mat",
        "/home/doujian/Desktop/Calib_Results.mat")


