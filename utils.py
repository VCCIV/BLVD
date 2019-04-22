from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
import argparse
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
            2]  # coordinate x,y,z   <====> length width height
        self.l = data[7]  # length
        self.w = data[8]  # width
        self.h = data[12] - data[11]  # height

        # TODO: Yaw angle (around Y-axis in LiDAR coordinates) [-pi..pi]

        # sin(yaw angle)
        self.sin_ry = data[9] / \
            np.sqrt(data[9] * data[9] + data[10] * data[10])
        # cos(yaw angle)
        self.cos_ry = data[10] / \
            np.sqrt(data[9] * data[9] + data[10] * data[10])
        print("-----------------------")
        print(self.sin_ry)
        print(self.cos_ry)

        if self.sin_ry >= 0:
            self.ry = np.arccos(data[10])
        else:
            self.ry = (-1) * np.arccos(data[10])

        print("rotate y-axis\t", np.rad2deg(self.ry))

        print("-----------------------")

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


def compute_box_corners_3d(object3d: Object3d) -> np.array:
    """Computes the 3D bounding box corner positions from an Object3d

    :param object3d: object3d to compute corners from
    :return: a numpy array of 3D corners if the box is in front of the camera,
             an empty array otherwise
    """

    # Compute rotational matrix
    """

    def roty(t):
        ''' Rotation along the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])


    """
    # coordinate x,y,z   <====> length width height
    rot = np.array([[+object3d.cos_ry, -object3d.sin_ry, 0],
                    [+object3d.sin_ry, +object3d.cos_ry, 0],
                    [0, 0, 1]])

    l = object3d.l
    w = object3d.w
    h = object3d.h

    # 3D Bounding Box Corners
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    z_corners = np.array([h / 2, h / 2, h / 2, h / 2, -
                          h / 2, -h / 2, -h / 2, -h / 2])

    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))
    # corners_3d = np.array([x_corners, y_corners, z_corners])

    corners_3d[0, :] = corners_3d[0, :] + object3d.t[0]
    corners_3d[1, :] = corners_3d[1, :] + object3d.t[1]
    corners_3d[2, :] = corners_3d[2, :] + object3d.t[2]

    # print(corners_3d)

    return np.transpose(corners_3d)


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

    print(np.max(imgfov_pc_rect[:, 2]), np.min(imgfov_pc_rect[:, 2]))

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


"""" Simplest drawing LiDAR Data """


def draw_lidar_simple(pc_data, color=None):
    """
     Draw lidar points. simplest set up.
    :param pc_data: input point cloud
    :param color:
    :return:
    """
    if "mlab" not in sys.modules:
        try:
            import mayavi.mlab as mlab
        except BaseException:
            print("mlab module should be installed")
            return

    fig = mlab.figure(
        figure=None, bgcolor=(
            0, 0, 0), fgcolor=None, engine=None, size=(
            1600, 1000))
    if color is None:
        color = pc_data[:, 2]
    # draw points
    mlab.points3d(pc_data[:, 0], pc_data[:, 1], pc_data[:, 2], color, color=None, mode='point', colormap='gnuplot', scale_factor=1,
                  figure=fig)
    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]],
                color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]],
                color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]],
                color=(0, 0, 1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0,
              figure=fig)
    return fig


def draw_lidar(pc, color=None, fig=None, bgcolor=(0, 0, 0),
               pts_scale=1, pts_mode='point', pts_color=None):
    """
     Draw lidar points
    :param pc: numpy array (n,3) of XYZ
    :param color: numpy array (n) of intensity or whatever
    :param fig: mayavi figure handler, if None create new one otherwise will use it
    :param bgcolor:
    :param pts_scale:
    :param pts_mode:
    :param pts_color:
    :return: created or used fig
    """

    if "mlab" not in sys.modules:
        try:
            import mayavi.mlab as mlab
        except BaseException:
            print("mlab module should be installed")
            return

    if fig is None:
        fig = mlab.figure(
            figure=None,
            bgcolor=bgcolor,
            fgcolor=None,
            engine=None,
            size=(
                1600,
                1000))
    if color is None:
        color = pc[:, 2]
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=pts_color, mode=pts_mode, colormap='gnuplot',
                  scale_factor=pts_scale, figure=fig)

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

    # draw axis
    axes = np.array([
        [2., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 2., 0.],
    ], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]],
                color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]],
                color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]],
                color=(0, 0, 1), tube_radius=None, figure=fig)

    # draw fov Todo: update to real sensor spec.
    fov = np.array([  # 45 degree
        [20., 20., 0., 0.],
        [20., -20., 0., 0.],
    ], dtype=np.float64)

    mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)
    mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
                figure=fig)

    # draw square region
    TOP_Y_MIN = -20
    TOP_Y_MAX = 20
    TOP_X_MIN = 0
    TOP_X_MAX = 40
    TOP_Z_MIN = -2.0
    TOP_Z_MAX = 0.4

    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5),
                tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5),
                tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5),
                tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5),
                tube_radius=0.1, line_width=1, figure=fig)

    # mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=70, focalpoint=[
              12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)

    return fig


def rotx(t):
    ''' 3D Rotation along the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    ''' Rotation along the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    ''' Rotation along the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

# Todo: draw 3d box in image and lidar
# Test 1


def draw_rgb_projections(image, projections, color=(
        255, 255, 255), thickness=2, darker=1):

    img = image.copy() * darker
    num = len(projections)
    forward_color = (255, 255, 0)
    for n in range(num):
        qs = projections[n]
        for k in range(0, 4):
            i, j = k, (k + 1) % 4

            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness)

        cv2.line(img, (qs[3, 0], qs[3, 1]), (qs[7, 0], qs[7, 1]),
                 forward_color, thickness)
        cv2.line(img, (qs[7, 0], qs[7, 1]), (qs[6, 0], qs[6, 1]),
                 forward_color, thickness)
        cv2.line(img, (qs[6, 0], qs[6, 1]), (qs[2, 0], qs[2, 1]),
                 forward_color, thickness)
        cv2.line(img, (qs[2, 0], qs[2, 1]), (qs[3, 0], qs[3, 1]),
                 forward_color, thickness)
        cv2.line(img, (qs[3, 0], qs[3, 1]), (qs[6, 0], qs[6, 1]),
                 forward_color, thickness)
        cv2.line(img, (qs[2, 0], qs[2, 1]), (qs[7, 0], qs[7, 1]),
                 forward_color, thickness)

    return img


# Test2

def draw_projected_box3d(image, qs, color=(255, 255, 255), thickness=1):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref:
        # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                               qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                               qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                               qs[j, 1]), color, thickness)
    return image


# Test 3


# Todo: transform 3d box to 2d box in image


def project_to_image_space(object3d, calib_data,
                           image_size=None):
    """

    :param box_3d: Single box_3d to project
    :param calib_data: Calibration object to be used
    :param image_size: [w, h] should be provided
    :return: Projected box in image space [x1, y1, x2, y2]
            Returns None if box is not inside the image
    """

    corners_3d = compute_box_corners_3d(object3d)
    projected = calib_data.project_velo_to_image(corners_3d)

    x1 = np.amin(projected[:, 0])
    y1 = np.amin(projected[:, 1])
    x2 = np.amax(projected[:, 0])
    y2 = np.amax(projected[:, 1])
    img_box = np.array([x1, y1, x2, y2])

    if not image_size:
        raise ValueError('Image size must be provided')

    image_w = image_size[0]
    image_h = image_size[1]

    # if img_box[0] > image_w or \
    #         img_box[1] > image_h or \
    #         img_box[2] < 0 or \
    #         img_box[3] < 0:
    #     return None

    img_box[0] = 0 if img_box[0] < 0 else img_box[0]
    img_box[1] = 0 if img_box[1] < 0 else img_box[1]

    img_box[2] = image_w if img_box[2] > image_w else img_box[2]
    img_box[3] = image_h if img_box[3] > image_h else img_box[3]

    img_box_w = img_box[2] - img_box[0]
    img_box_h = img_box[3] - img_box[1]

    if img_box_w > (image_w * 0.8) and img_box_h > (image_h * 0.8):
        return None

    return corners_3d, projected, img_box


if __name__ == "__main__":

    # test 000001

    """Annotation"""
    label_filename = "/home/doujian/Desktop/Dataset/label/000050.txt"
    objects = read_label(label_filename)

    print("test compute box corners 3d")

    compute_box_corners_3d(objects[0])

    """ Calibration """
    calib = Calibration(
        "/home/doujian/Desktop/Rt.mat",
        "/home/doujian/Desktop/Calib_Results.mat")
    """ 3D LiDAR """
    # Todo: Use matlab function (MatlabFunctionForLiDAR) to transform the .txt file to .bin file
    # Todo: Reference: https://github.com/DrGabor/LiDAR
    lidar = load_velo_scan("/home/doujian/Desktop/Dataset/lidar/000050.bin")

    velo_data = lidar[:, :3]

    print(np.max(velo_data[:, 0]), np.min(velo_data[:, 0]))  # length  x
    print(np.max(velo_data[:, 1]), np.min(velo_data[:, 1]))  # width   y
    print(np.max(velo_data[:, 2]), np.min(velo_data[:, 2]))  # height  z

    lidar_to_img = calib.project_velo_to_image(velo_data)

    print(lidar_to_img.shape)

    print(lidar_to_img[:, :10])
    """ Image """
    img = load_image("/home/doujian/Desktop/Dataset/image/50.jpg")

    # show_lidar_on_image(velo_data, img, calib)

    for single_object in objects:
        corner_3d, projected_3d, single_pts2d = project_to_image_space(
            single_object, calib, [
                img.shape[1], img.shape[0]])
        print(single_pts2d)
        if single_pts2d is not None:
            top_left = (int(single_pts2d[0]), int(single_pts2d[1]))
            down_right = (int(single_pts2d[2]), int(single_pts2d[3]))
            cv2.rectangle(img, top_left, down_right, (255, 0, 0), 2)
            # draw_projected_box3d(img, projected_3d)

    cv2.imshow("img", img)
    cv2.waitKey(0)
