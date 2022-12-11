import warnings
from scipy.optimize import curve_fit
import numpy as np
import cv2

def draw_edge(im, x, y, bw=1, color=(255, 255, 255), draw_end_points=False):
    r"""Set colors given a list of x and y coordinates for the edge.

    Args:
        im (HxWxC numpy array): Canvas to draw.
        x (1D numpy array): x coordinates of the edge.
        y (1D numpy array): y coordinates of the edge.
        bw (int): Width of the stroke.
        color (list or tuple of int): Color to draw.
        draw_end_points (bool): Whether to draw end points of the edge.
    """
    if x is not None and x.size:
        h, w = im.shape[0], im.shape[1]
        # Draw edge.
        for i in range(-bw, bw):
            for j in range(-bw, bw):
                yy = np.maximum(0, np.minimum(h - 1, y + i))
                xx = np.maximum(0, np.minimum(w - 1, x + j))
                set_color(im, yy, xx, color)

        # Draw endpoints.
        if draw_end_points:
            for i in range(-bw * 2, bw * 2):
                for j in range(-bw * 2, bw * 2):
                    if (i ** 2) + (j ** 2) < (4 * bw ** 2):
                        yy = np.maximum(0, np.minimum(h - 1, np.array(
                            [y[0], y[-1]]) + i))
                        xx = np.maximum(0, np.minimum(w - 1, np.array(
                            [x[0], x[-1]]) + j))
                        set_color(im, yy, xx, color)


def set_color(im, yy, xx, color):
    r"""Set pixels of the image to the given color.

    Args:
        im (HxWxC numpy array): Canvas to draw.
        xx (1D numpy array): x coordinates of the pixels.
        yy (1D numpy array): y coordinates of the pixels.
        color (list or tuple of int): Color to draw.
    """
    if type(color) != list and type(color) != tuple:
        color = [color] * 3
    if len(im.shape) == 3 and im.shape[2] == 3:
        if (im[yy, xx] == 0).all():
            im[yy, xx, 0], im[yy, xx, 1], im[yy, xx, 2] = \
                color[0], color[1], color[2]
        else:
            for c in range(3):
                im[yy, xx, c] = ((im[yy, xx, c].astype(float)
                                  + color[c]) / 2).astype(np.uint8)
    else:
        im[yy, xx] = color[0]


def interp_points(x, y):
    r"""Given the start and end points, interpolate to get a curve/line.

    Args:
        x (1D array): x coordinates of the points to interpolate.
        y (1D array): y coordinates of the points to interpolate.

    Returns:
        (dict):
          - curve_x (1D array): x coordinates of the interpolated points.
          - curve_y (1D array): y coordinates of the interpolated points.
    """
    if abs(x[:-1] - x[1:]).max() < abs(y[:-1] - y[1:]).max():
        curve_y, curve_x = interp_points(y, x)
        if curve_y is None:
            return None, None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if len(x) < 3:
                    popt, _ = curve_fit(linear, x, y)
                else:
                    popt, _ = curve_fit(func, x, y)
                    if abs(popt[0]) > 1:
                        return None, None
            except Exception:
                return None, None
        if x[0] > x[-1]:
            x = list(reversed(x))
            y = list(reversed(y))
        curve_x = np.linspace(x[0], x[-1], int(np.round(x[-1]-x[0])))
        if len(x) < 3:
            curve_y = linear(curve_x, *popt)
        else:
            curve_y = func(curve_x, *popt)
    return curve_x.astype(int), curve_y.astype(int)


def func(x, a, b, c):
    r"""Quadratic fitting function."""
    return a * x**2 + b * x + c


def linear(x, a, b):
    r"""Linear fitting function."""
    return a * x + b

def connect_face_keypoints(image_h, image_w, keypoints, add_upper_face=False, add_dist_map=False):
    r"""Connect the face keypoints to edges and draw the sketch.

    Args:
        image_h (int): Height the input image was resized to.
        image_w (int): Width the input image was resized to.
        keypoints (NxKx2 numpy array): Facial landmarks (with K keypoints).
        add_upper_face (bool)
        add_dist_map (bool)

    Returns:
        (list of HxWxC numpy array): Drawn label map.
    """
    # Mapping from keypoint index to facial part.
    part_list = [[range(0, 17)],
            [range(17, 22)],  # right eyebrow
            [range(22, 27)],  # left eyebrow
            [[28, 31], range(31, 36), [35, 28]],  # nose
            [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
            [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
            [range(48, 55), [54, 55, 56, 57, 58, 59, 48],
            range(60, 65), [64, 65, 66, 67, 60]],  # mouth and tongue
    ]
    
    if add_upper_face:
        pts = keypoints[:, :17, :].astype(np.int32)
        baseline_y = (pts[:, 0:1, 1] + pts[:, -1:, 1]) / 2
        upper_pts = pts[:, 1:-1, :].copy()
        upper_pts[:, :, 1] = baseline_y + (
                baseline_y - upper_pts[:, :, 1]) * 2 // 3
        keypoints = np.hstack((keypoints, upper_pts[:, ::-1, :]))

    edge_len = 3  # Interpolate 3 keypoints to form a curve when drawing edges.
    bw = max(1, image_h // 256)  # Width of the stroke.

    outputs = []
    for t_idx in range(keypoints.shape[0]):
        # Edge map for the face region from keypoints.
        im_edges = np.zeros((image_h, image_w, 1), np.uint8)
        im_dists = np.zeros((image_h, image_w, 0), np.uint8)
        for edge_list in part_list:
            for e, edge in enumerate(edge_list):
                # Edge map for the current edge.
                im_edge = np.zeros((image_h, image_w, 1), np.uint8)
                # Divide a long edge into multiple small edges when drawing.
                for i in range(0, max(1, len(edge) - 1), edge_len - 1):
                    sub_edge = edge[i:i + edge_len]
                    x = keypoints[t_idx, sub_edge, 0]
                    y = keypoints[t_idx, sub_edge, 1]

                    # Interp keypoints to get the curve shape.
                    curve_x, curve_y = interp_points(x, y)
                    draw_edge(im_edges, curve_x, curve_y, bw=bw)
                    if add_dist_map:
                        draw_edge(im_edge, curve_x, curve_y, bw=bw)

                if add_dist_map:
                    # Add distance transform map on each facial part.
                    im_dist = cv2.distanceTransform(255 - im_edge,
                                                    cv2.DIST_L1, 3)
                    im_dist = np.clip((im_dist / 3), 0, 255)
                    im_dists = np.dstack((im_dists, im_dist))

                # if add_pos_encode and e == 0:
                #     # Add positional encoding for the first edge.
                #     from math import pi
                #     im_pos = np.zeros((resize_h, resize_w, 0), np.float32)
                #     for l in range(10):  # noqa: E741
                #         dist = (im_dist.astype(np.float32) - 127.5) / 127.5
                #         sin = np.sin(pi * (2 ** l) * dist)
                #         cos = np.cos(pi * (2 ** l) * dist)
                #         im_pos = np.dstack((im_pos, sin, cos))

        # Combine all components to form the final label map.
        if add_dist_map:
            im_edges = np.dstack((im_edges, im_dists))
        im_edges = im_edges.astype(np.float32) / 255.0
        # if add_pos_encode:
        #     im_edges = np.dstack((im_edges, im_pos))
        outputs.append(im_edges)
    return outputs