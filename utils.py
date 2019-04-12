from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
import argparse
import math
import scipy.io as sio
""" Display """
import matplotlib.pyplot as plt


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
        # Tr_velo_to_cam[:, 1] = Tr_velo_to_cam[:, 1] * (-1)

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

    def cart2hom(self, pts_3d):
        """

        :param pts_3d: n*3 points in Cartesian
        :return: n*4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================

    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo_hom = self.cart2hom(pts_3d_velo)  # n*4
        return np.dot(pts_3d_velo_hom, np.transpose(self.Tr_velo_to_cam))

    def project_ref_to_rect(self, pts_3d_ref):
        """
        Input and Output are nx3 points (same as KITTI Dataset)
        :param pts_3d_ref:
        :return:
        """
        return np.transpose(np.dot(self.ro_rect, np.transpose(pts_3d_ref)))

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        """

        :param pts_3d_rect: n*3 points in rect camera coordinate.
        :return: n*2 points in image coordinate.
        """

        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.cam_matrix_p))  # n*3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]

        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        """
        :param pts_3d_velo: nx3 points in velodyne coordinate.
        :return: nx2 points in image coordinate.
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)


def get_lidar_in_image_fov(pts_3d, calib, xmin, ymin,
                           xmax, ymax, clip_distance=2.0):
    """

    :param pts_3d: 3d Point Cloud
    :param calib: Calibration class
    :param xmin:
    :param ymin:
    :param xmax:
    :param ymax:
    :param clip_distance:
    :return: imgfov_pc_3d(lidar data in image fov), pts_2d(lidar data in image coordinate), fov_inds
    """
    pts_2d = calib.project_velo_to_image(pts_3d)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & (
        pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pts_3d[:, 0] > clip_distance)
    imgfov_pc_3d = pts_3d[fov_inds, :]

    return imgfov_pc_3d, pts_2d, fov_inds


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def load_image(img_filename):
    img = cv2.imread(img_filename)
    # img = img[..., ::-1]
    return img


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float64)
    scan = scan.reshape((-1, 4))
    return scan


def show_lidar_on_image(pc_data, img, calib):
    """
     Project LiDAR points to image
    :param pc_data:
    :param img:
    :param calib:
    :return:
    """
    img_height, img_width, _ = img.shape
    imgfov_pc_3d, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_data, calib, 0, 0, img_width, img_height)

    imgfov_pts_2d = pts_2d[fov_inds, :]

    print("point cloud in img fov:\t", imgfov_pts_2d.shape)
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_3d)

    print(np.max(imgfov_pc_rect[:,2]),np.min(imgfov_pc_rect[:,2]))

    cmap = plt.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]

        color = cmap[int(512.0 / depth), :]
        # color = cmap[128, :]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i, 0])),
                         int(np.round(imgfov_pts_2d[i, 1]))),
                   1, color=tuple(color), thickness=-1)
    # Image.fromarray(img).show()
    cv2.imshow("img", img)
    cv2.waitKey(0)
    return img


if __name__ == "__main__":

    # test 000001
    """Annotation"""
    label_filename = "/home/doujian/Desktop/Dataset/label/000001.txt"
    objects = read_label(label_filename)
    """ Calibration """
    calib = Calibration(
        "/home/doujian/Desktop/Rt.mat",
        "/home/doujian/Desktop/Calib_Results.mat")
    """ 3D LiDAR """
    #Todo: Use matlab function to transform the .txt file to .bin file
    #Todo: Reference: https://github.com/DrGabor/LiDAR
    lidar = load_velo_scan("/home/doujian/Desktop/Dataset/lidar/000001.bin")

    velo_data = lidar[:, :3]

    print(np.max(velo_data[:, 0]), np.min(velo_data[:, 0]))  # length  x
    print(np.max(velo_data[:, 1]), np.min(velo_data[:, 1]))  # width   y
    print(np.max(velo_data[:, 2]), np.min(velo_data[:, 2]))  # height  z

    lidar_to_img = calib.project_velo_to_image(velo_data)

    print(lidar_to_img.shape)

    print(lidar_to_img[:, :10])
    """ Image """
    img = load_image("/home/doujian/Desktop/Dataset/image/1.jpg")

    show_lidar_on_image(velo_data, img, calib)
