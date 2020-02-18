import numpy as np
import cv2
import os


def show(img):
    """
    Show input image in a window.

    Parameters
    ----------
    img
        np.array, image opened with OpenCV

    Returns
    -------
    None
    """
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
