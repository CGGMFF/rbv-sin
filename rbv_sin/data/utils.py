
from typing import Tuple, Sequence
import numpy as np
import cv2

def meanMask(masks : Sequence[np.ndarray], centre : Tuple[float, float] = None, samples : int = 360, verbose : int = 0) -> np.ndarray:
    if centre is None:
        mask_sum = np.sum(masks, axis=0)
        intersection = mask_sum == len(masks)
        rr, cc = np.mgrid[0 : mask_sum.shape[0], 0 : mask_sum.shape[1]]
        c_r, c_c = np.mean(rr[intersection]), np.mean(cc[intersection])
    else:
        c_r, c_c = centre
    centre = np.c_[np.asarray([c_r, c_c])]

    def newPoints(vectors, length, centre):
        contour_points = vectors * length + centre
        contour_points_int = np.asarray(contour_points, int)
        return contour_points, contour_points_int

    angles = np.linspace(0, 2 * np.pi, samples)
    vectors = np.asarray([np.sin(angles), np.cos(angles)])
    current_length = 10
    contour_points, contour_points_int = newPoints(vectors, current_length, centre)
    
    lengths = np.ones([len(masks), samples])
    in_mask = np.ones([len(masks), samples])

    while np.any(in_mask):
        current_length += 1
        contour_points, contour_points_int = newPoints(vectors, current_length, centre)
        for idx, mask in enumerate(masks):
            current_in_mask = mask[contour_points_int[0], contour_points_int[1]]
            lengths[idx, current_in_mask] = current_length - 1 # To select a point still inside the mask
            in_mask[idx] = current_in_mask

    mean_lengths = np.mean(lengths, axis=0)
    final_points, final_points_int = newPoints(vectors, mean_lengths, centre)
    hull = cv2.convexHull(final_points_int.T[:, [1, 0]])
    mask = np.zeros(masks[0].shape[:2], dtype=np.uint8)
    mask = cv2.fillPoly(mask, [hull], 1)
    return mask
