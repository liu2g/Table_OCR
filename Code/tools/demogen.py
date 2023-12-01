import cv2
import imutils
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import skimage
import seaborn as sns

color_g = (102, 181, 129)
color_b = (105, 171, 201)
alpha = 0.5


def show_img(img):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    y, x, *_ = img.shape
    ax.set_xticks([0, int(x / 2), x])
    ax.set_yticks([0, int(y / 2), y])
    ax.set_yticklabels(["", int(y / 2), y])
    return fig, ax


def demo_vanilla(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    img_bin = cv2.dilate(img_bin, None, iterations=1)
    contours = cv2.findContours(img_bin.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    img_decorated = img.copy()
    cv2.drawContours(img_decorated, contours, -1, color_b, 2)
    img_decorated = cv2.addWeighted(img_decorated, alpha, img, 1 - alpha, 0)
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(img_decorated, (x, y), (x + w, y + h), color_g, 2)
    fig, ax = show_img(img_decorated)
    fig.savefig(str(Path.home() / "vanilla_demo.pdf"))

def demo_conncomp(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_bin = cv2.erode(img_bin, np.ones((2, 2), np.uint8),
                          iterations=1)
    img_bin = cv2.dilate(img_bin, np.ones((3, 3), np.uint8),
                           iterations=2)
    img_bin = skimage.morphology.area_opening(img_bin)
    bounding_boxes = []
    analysis = cv2.connectedComponentsWithStats(img_bin, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    img_mask = label_ids > 0
    # Loop through each component
    img_decorated = img.copy()
    img_decorated[img_mask] = color_b
    img_decorated = cv2.addWeighted(img_decorated, alpha, img, 1-alpha, 0)
    for i in range(1, totalLabels):
        # Area of the component
        area = values[i, cv2.CC_STAT_AREA]

        # Now extract the coordinate points
        x1 = values[i, cv2.CC_STAT_LEFT]
        y1 = values[i, cv2.CC_STAT_TOP]
        w = values[i, cv2.CC_STAT_WIDTH]
        h = values[i, cv2.CC_STAT_HEIGHT]

        bounding_boxes.append([x1, y1, w, h])

        # Coordinate of the bounding box
        pt1 = (x1, y1)
        pt2 = (x1 + w, y1 + h)

        # Bounding boxes for each component
        cv2.rectangle(img_decorated, pt1, pt2, color_g, 2)
    fig, ax = show_img(img_decorated)
    fig.savefig(str(Path.home() / "conncomp_demo.pdf"))


def demo_histsplit(img):
    sns.set_theme()
    y, x, *_ = img.shape
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = image_gray.shape
    gs = plt.GridSpec(2, 2, width_ratios=[2, 8], height_ratios=[2, 8])
    fig = plt.figure(figsize=(10, h/w*10))
    ax_void = fig.add_subplot(gs[0, 0], adjustable='datalim')
    ax_void.axis("off")
    def remove_axis_annotate(ax):
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
    ax_col = fig.add_subplot(gs[0, 1], adjustable='datalim', sharey=ax_void)
    ax_col.set_xlim([0, w])
    remove_axis_annotate(ax_col)
    ax_row = fig.add_subplot(gs[1, 0], adjustable='datalim', sharex=ax_void)
    remove_axis_annotate(ax_row)
    ax_row.set_ylim([h, 0])
    ax_img = fig.add_subplot(gs[1, 1], aspect=h/w)
    remove_axis_annotate(ax_img)
    _, image_bin = cv2.threshold(image_gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    image_bin = cv2.erode(image_bin, np.ones((2, 2), np.uint8),
                          iterations=1)
    image_bin = cv2.dilate(image_bin, np.ones((3, 3), np.uint8),
                           iterations=1)
    image_bin = skimage.morphology.area_opening(image_bin)
    sum_each_row = np.sum(image_bin > 0, axis=1)
    sum_each_column = np.sum(image_bin > 0, axis=0)
    ax_row.plot(1 - sum_each_row / np.max(sum_each_row), np.arange(h))
    ax_col.plot(np.arange(w), sum_each_column / np.max(sum_each_row))
    xbounds = []
    bound = []
    for i, r in enumerate(sum_each_row):
        if (not bound) and r:  # start of bound
            bound.append(i)
        elif bound and (not r or i == len(sum_each_row) - 1):  # end of bound
            bound.append(i)
            xbounds.append(bound)
            bound = []
    xbounds = np.array(xbounds)
    image_decorated = img.copy()
    boxes = []
    box = []
    for bound in xbounds:
        bound_upper, bound_lower = bound
        ax_img.axhline(bound_lower, color=np.array(color_b) / 255)
        ax_img.axhline(bound_upper, color=np.array(color_b) / 255)
        image_band = image_bin[bound_upper:bound_lower]
        sum_each_col = np.sum(image_band > 0, axis=0)
        for i, c in enumerate(sum_each_col):
            if (not box) and c:  # start of bound
                ax_img.axvline(i, color=np.array(color_b) / 255)
                box.extend([bound_upper, i])
            elif box and (not c or i == len(sum_each_col) - 1):  # end of bound
                ax_img.axvline(i, color=np.array(color_b) / 255)
                box.extend([bound_lower - bound_upper, i - box[-1]])
                boxes.append(box)
                box = []
    boxes = np.array(boxes)
    for box in boxes:
        x, y, h, w = box
        overlay = image_decorated.copy()
        cv2.rectangle(overlay, (y, x), (y + w, x + h), color_g,
                      -1)
        image_decorated = cv2.addWeighted(overlay, alpha, image_decorated,
                                          1 - alpha, 0)
    ax_img.imshow(image_decorated, cmap="gray", vmin=0, vmax=255)
    fig.savefig(str(Path.home() / "histsplit_demo.pdf"))

if __name__ == "__main__":
    img_file = Path.home() / "Datasets/bboxtestimg/demo.jpg"
    img = cv2.imread(str(img_file))
    demo_vanilla(img)
    demo_conncomp(img)
    demo_histsplit(img)

