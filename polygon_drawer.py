import numpy as np
import cv2
import copy
import os
import math

CANVAS_SIZE = (600, 800)
FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)


def load_images_from_folder(folder):
    images = []
    img_names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            img_names.append(filename)
    # converting to numpy type array
    images = np.array(images)
    return images, img_names


class PolygonDrawer(object):
    def __init__(self, window_name, image):
        self.window_name = window_name  # Name for our window
        self.image = image
        self.done = False  # Flag signalling we're done
        # Current position, so we can draw the line-in-progress
        self.current = (0, 0)
        self.points = []  # List of points defining our polygon

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done:  # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" %
                  (len(self.points), x, y))

            ####################################
            # can put distance related checks here
            ####################################

            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        # cv2.namedWindow(self.window_name, flags=cv2.CV_WINDOW_AUTOSIZE)
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            #             canvas = np.zeros(CANVAS_SIZE, np.uint8)
            canvas = self.image.copy()
            # if (len(self.points) > 0):
            # Draw all the current polygon segments
            # cv2.polylines(canvas, np.array(
            #     [self.points]), False, FINAL_LINE_COLOR, 1)
            # # And  also show what the current segment would look like
            # cv2.line(canvas, self.points[-1],
            #          self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            # if cv2.waitKey(50) == 27:  # ESC hit
            if cv2.waitKey(50) == ord(" "):  # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing

        cv2.destroyAllWindows()
        H, W, _ = canvas.shape
        mask = np.zeros((H, W), np.uint8)
        # of a filled polygon
        if (len(self.points) > 0):
            # cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
            cv2.fillPoly(mask, np.array([self.points]), FINAL_LINE_COLOR)
        # And show it
        # cv2.imshow(self.window_name, canvas)
        # cv2.imshow("mask", mask)
        # Waiting for the user to press any key
        # cv2.imwrite("masked_roi_road.jpg", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return mask, self.points


def distance_euclidean(p1=(0, 0), p2=(0, 0)):
    """calculate eucleadean distance of bboxe centers of two consecutive
    frames and appends in global variable of DIST

    Parameters
    ----------
    p1 : tuple
        point 1 (x1,y1)
    p2 : tuple
        point 2 (x2,y2)

    Returns
    -------
    NILL
    """
    # d = \sqrt{(x_2 - x_1)^2 + (y_2-y_1)^2}
    (x1, y1) = p1
    (x2, y2) = p2
    return (math.sqrt(((x2-x1)**2) + ((y2-y1)**2)))


if __name__ == "__main__":

    # all images as numpy matrix
    imgs, imgs_names = load_images_from_folder("input/")
    # print("imgs", imgs)
    # print("imgs_names = ", imgs_names)
    for i, images in enumerate(imgs):
        pd = PolygonDrawer("Polygon", images)
        masked, points = pd.run()
        # cv2.imwrite("demo.jpg", image_original)
        cv2.imwrite("output/{}.jpg".format(imgs_names[i]), masked)
        # print("Polygon = %s" % pd.points)
