from PIL import Image
import numpy as np
import cv2
import copy
import os

# CANVAS_SIZE = (10, 10)
FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)

PADDING_PIXELS = 50


def load_images_from_folder(folder):
    images = []
    img_names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, (256, 256))
        top, bottom, left, right = (
            PADDING_PIXELS, PADDING_PIXELS, PADDING_PIXELS, PADDING_PIXELS)
        padded_im = cv2.copyMakeBorder(
            img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if img is not None:
            # images.append(img)
            images.append(padded_im)
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
        # cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
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
            # cv2.resizeWindow(self.window_name, 60, 80)
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            # if cv2.waitKey(50) == 27:  # ESC hit
            key = cv2.waitKey(33)
            # 32 == space bar
            if key == 32:
                self.done = True
            # 'q' is presssed delete last point
            elif key == 113:
                if len(self.points) > 0:
                    print("removing last line point")
                    del self.points[-1]

        # User finised entering the polygon points, so let's make the final drawing

        cv2.destroyAllWindows()
        H, W, _ = canvas.shape
        mask = np.zeros((H, W), np.uint8)
        # of a filled polygon
        if (len(self.points) > 0):
            # cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
            cv2.fillPoly(mask, np.array([self.points]),
                         FINAL_LINE_COLOR, lineType=cv2.LINE_AA)
        # And show it
        # cv2.imshow(self.window_name, canvas)
        # cv2.imshow("mask", mask)
        # Waiting for the user to press any key
        # cv2.imwrite("masked_roi_road.jpg", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.points = []
        return mask, canvas, self.points


def create_collab(_img1, _img2, _path):
    # images = [Image.open(x) for x in ['logo.png', 'logo.png']]
    _img1 = im = cv2.cvtColor(_img1, cv2.COLOR_BGR2RGB)
    _img2 = im = cv2.cvtColor(_img2, cv2.COLOR_BGR2RGB)

    images = [Image.fromarray(_img2), Image.fromarray(_img1)]

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_im.save(_path)


if __name__ == "__main__":

    # folder of folders
    inp_folder = "input/"

    if not os.path.exists("output/"):
        os.mkdir("output")
    for folders in os.listdir(inp_folder):
        # print(folders)
        imgs, imgs_names = load_images_from_folder(inp_folder+folders)
        # print("imgs", imgs.shape)
        # print("names ", imgs_names)
        # print("imgs_names = ", imgs_names)
        for i, images in enumerate(imgs):
            print("image ", inp_folder + folders + "/" + imgs_names[i])

            if os.path.exists("output/{}/{}".format(folders, imgs_names[i])):
                print(
                    "output/{}/{} already exists".format(folders, imgs_names[i]))
                continue

            pd = PolygonDrawer("{}/{}".format(folders, imgs_names[i]), images)
            masked, original, points = pd.run()

            # create output folder if not exists
            if not os.path.exists("output/{}".format(folders)):
                os.makedirs("output/{}/".format(folders))

            masked = masked[PADDING_PIXELS:PADDING_PIXELS +
                            256, PADDING_PIXELS:PADDING_PIXELS+256]
            original = original[PADDING_PIXELS:PADDING_PIXELS +
                                256, PADDING_PIXELS:PADDING_PIXELS+256]
            # masked = cv2.resize(masked, (256, 256))
            # original = cv2.resize(original, (256, 256))
            create_collab(masked, original,
                          "output/{}/{}".format(folders, imgs_names[i]))
