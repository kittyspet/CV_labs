import numpy as np
import cv2 as cv


def rotate(img: np.ndarray, pivot: tuple, angle: float) -> np.ndarray:
    img_corners = np.array([[img.shape[1] - 1, img.shape[0] - 1],
                             [0, 0], [img.shape[1] - 1, 0],
                             [0, img.shape[0] - 1]])

    rotation_mat = cv.getRotationMatrix2D(pivot, angle, scale=1.0)
    new_dimensions = compute_new_size(img_corners, rotation_mat)

    return cv.warpAffine(img, rotation_mat, new_dimensions)


def apply_warpAffine(img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    img_corners = np.array([[img.shape[1] - 1, img.shape[0] - 1],
                             [0, 0], [img.shape[1] - 1, 0],
                             [0, img.shape[0] - 1]])

    affine_mat = cv.getAffineTransform(src_pts, dst_pts)
    new_dimensions = compute_new_size(img_corners, affine_mat)

    return cv.warpAffine(img, affine_mat, new_dimensions)


def compute_new_size(corners: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    homogeneous_coords = np.hstack((corners, np.ones((corners.shape[0], 1))))
    transformed_coords = transform_matrix @ homogeneous_coords.T

    min_coords = transformed_coords.min(axis=1)
    max_coords = transformed_coords.max(axis=1)

    transform_matrix[:, 2] -= min_coords
    new_size = max_coords - min_coords
    new_size = np.int64(np.ceil(new_size))

    return new_size