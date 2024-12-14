
from typing import Tuple

import numpy as np
import cv2

from rbv_sin.data import RFI

class AABBSelector:
    """
    This class represents a utility for selecting rectangular regions of interest from retinal fundus images
    but also provides functions to process general images as well.
    The class uses OpenCV for event processing and image rendering.
    """

    def __init__(self, example_size : int = 800) -> None:
        """
        Stores the requested size of the displayed images. This is necessary because OpenCV doesn't
        resize window automatically.

        Arguments:
        - 'example_size' - The height of the image in pixels shown to the user.
        """
        self.example_size = example_size

    def getAABB(self, rfi : RFI) -> Tuple[int, int, int, int]:
        """
        Queries the user for an AABB from a retinal fundus image. It shows the 'rfi.image' attribute to
        the user and returns 4 integers representing the boundaries of the rectangle.

        Arguments:
        - 'rfi' - 'RFI' object with defined 'image' attribute.

        Returns:
        - 'x1', 'y1', 'x2', 'y2' coordinates of the selected rectangle (y - rows, x - columns).
        """
        x1, y1, x2, y2 = self.select(rfi.image)
        return x1, y1, x2, y2

    def select(self, image : np.ndarray) -> Tuple[int, int, int, int]:
        """
        Initialises the AABB selection interface and allows the user to draw a rectangle using left
        mouse button drag and confirmation with right mouse button click.
        Only the last drawn rectangle is confirmed, so the user may redraw the bounding rectangle
        as many times as s/he wants.

        The function resizes the image to the size specified in the constructor before showing it
        to the user. This happens to prevent creating windows larger than the user's screen.

        Arguments:
        - 'image' - Image which will be shown to the user.

        Returns:
        - 'x1', 'y1', 'x2', 'y2' coordinates of the selected rectangle (y - rows, x - columns).
        """
        rectangles = []
        rectangle = None
        end = False
        start = None
        finish = None

        ratio = image.shape[0] / self.example_size
        image = cv2.resize(image, (int(image.shape[1] / ratio), int(image.shape[0] / ratio)))

        def mouseCallback(event, x, y, flags, param):
            nonlocal rectangles, rectangle, start, end, finish
            if event == cv2.EVENT_LBUTTONDOWN:
                start = (x, y)
                end = None
            if event == cv2.EVENT_LBUTTONUP:
                rectangles.append((start, (x, y)))
                rectangle = (start, (x, y))
                start = None
                end = None
            if event == cv2.EVENT_MOUSEMOVE:
                if start is not None:
                    end = (x, y)
            if event == cv2.EVENT_RBUTTONUP:
                finish = True

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouseCallback)

        canvas = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        if canvas.ndim < 3:
            canvas = np.dstack([canvas, canvas, canvas])

        while True:
            drawn = canvas.copy()
            if end is not None and start is not None:
                cv2.rectangle(drawn, (start[0], start[1]), (end[0], end[1]), (0, 0, 255), 2)
            if rectangle is not None:
                cv2.rectangle(drawn, (rectangle[0][0], rectangle[0][1]), (rectangle[1][0], rectangle[1][1]), (0, 255, 0), 2)

            cv2.imshow("image", drawn)
            key = cv2.waitKey(1) & 0xFF

            if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == 27:
                points = []
                hover = None
            if finish:
                break

        cv2.destroyAllWindows()

        return int(rectangle[0][0] * ratio), int(rectangle[0][1] * ratio), int(rectangle[1][0] * ratio), int(rectangle[1][1] * ratio)
