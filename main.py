import argparse
import cv2
import imutils
import numpy as np

from imutils.video import FileVideoStream

def main(imagedir):

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    font_color = (0, 0, 255)
    font_thickness = 2

    vidcap = FileVideoStream(imagedir).start()

    title_font_size = 1.0

    while vidcap.more():

        grade = ''
        dead_knot = False
        small_knot = False
        holes = 0
        cracks = False

        # frame acquisition
        frame = vidcap.read()
        if frame is None:
            break

        # frame preprocessing
        frame = frame[:, 900:2500]
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # foreground separation
        wood_mask = foreground_separation(hsv)
        wood_mask_kernel = np.ones((9, 9), np.uint8)
        dilated_mask = cv2.morphologyEx(wood_mask, cv2.MORPH_ERODE, wood_mask_kernel, iterations=2)

        # wood v channel but masked, and gamma corrected
        wood_darkness = cv2.bitwise_and(gray, gray, mask=wood_mask)
        wood_darkness = automatic_gamma_adjustment(wood_darkness, wood_mask, intended_gamma=1.8)

        # get thresholded img with features
        holes_img, crack_img, knot_img = find_defects(wood_darkness, dilated_mask)
        holes_img = cv2.bitwise_and(holes_img, cv2.bitwise_not(knot_img))

        # find contours and edges
        knot_cnts, _ = cv2.findContours(knot_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        holes_cnts, _ = cv2.findContours(holes_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        wood_cnts, _ = cv2.findContours(wood_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        crack_edges = imutils.auto_canny(crack_img, sigma=0.25)

        # display all defects
        lines = cv2.HoughLinesP(crack_edges, 1, np.pi/180, 60, minLineLength=50, maxLineGap=60)
        if lines is not None:
            cracks = True
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, 'Crack', (x1, y1), font, font_size, (255, 255, 255), font_thickness)

        for c in knot_cnts:
            area = cv2.contourArea(c)
            (x, y, w, h) = cv2.boundingRect(c)
            if area < 6000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Small Knot', (x, y), font, font_size, font_color, font_thickness)
                small_knot = True
            elif area > 1000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, 'Dead Knot', (x, y), font, font_size, font_color, font_thickness)
                dead_knot = True

        for c in holes_cnts:
            area = cv2.contourArea(c)
            if area < 200:
                ((x, y), r) = cv2.minEnclosingCircle(c)
                cv2.circle(frame, (int(x), int(y)), int(r), (255, 0, 0), 1)
                cv2.putText(frame, 'Hole', (int(x), int(y)), font, font_size, (255, 0, 0), font_thickness)
                holes += 1

        for c in wood_cnts:
            area = cv2.contourArea(c)
            if area > 20000:
                cv2.drawContours(frame, wood_cnts, -1, (0,0,255), 1)


        # display grade
        if dead_knot == True or cracks == True:
            grade = 'C'
        elif holes > 7:
            grade = 'C'
        elif holes > 0:
            grade = 'B'
        elif holes == 0 and small_knot == True:
            grade = 'B'
        else:
            grade = 'A'

        cv2.putText(frame, 'Grade: ' + grade, (5, 30), font, title_font_size, font_color, font_thickness)
        cv2.putText(frame, 'Dead Knot: ' + str(dead_knot), (5, 60), font, title_font_size, font_color, font_thickness)
        cv2.putText(frame, 'Small Knot: ' + str(small_knot), (5, 90), font, title_font_size, font_color, font_thickness)
        cv2.putText(frame, 'Cracks: ' + str(cracks), (5, 120), font, title_font_size, font_color, font_thickness)
        cv2.putText(frame, 'Holes: ' + str(holes), (5, 150), font, title_font_size, font_color, font_thickness)

        cv2.imshow("Video", frame)

        if cv2.waitKey(50) == 27:
            break
        elif cv2.waitKey(50) == 112:
            cv2.waitKey(-1)

    cv2.destroyAllWindows()


def foreground_separation(hsv):
    # define range of wood color in HSV
    lower_wood = np.array([0, 5, 20])
    upper_wood = np.array([30, 255, 255])

    # Threshold the HSV image to get only wood colors
    mask = cv2.inRange(hsv, lower_wood, upper_wood)

    # opening and closing
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    return mask


def gamma_adjustment(img, gamma = 1.0):
    table = np.array([((i / 255.0) ** gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def automatic_gamma_adjustment(img, mask, intended_gamma = 2.2):
    mean = cv2.mean(img, mask=mask)
    gamma = intended_gamma / (1 / (mean[0] / 255))
    return gamma_adjustment(img, gamma)


def find_defects(wood_darkness, dilated_mask):
    
    # apply clahe
    gray = gamma_adjustment(wood_darkness, 0.74074)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(32, 32))
    gray = clahe.apply(gray)

    # apply threshold
    _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
    img = cv2.bitwise_and(thresh, thresh, mask=dilated_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    holes_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel = np.ones((3, 3), np.uint8)
    crack_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    

    # knot has a special clahe requirement
    knot_gray = gamma_adjustment(wood_darkness, 0.83333)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    knot_gray = clahe.apply(knot_gray)

    # apply threshold (knots only)
    _, thresh = cv2.threshold(knot_gray, 90, 255, cv2.THRESH_BINARY_INV)
    k_img = cv2.bitwise_and(thresh, thresh, mask=dilated_mask)
    
    knot_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    knot_img = cv2.morphologyEx(k_img, cv2.MORPH_OPEN, knot_kernel, iterations=3)
    knot_img = cv2.morphologyEx(knot_img, cv2.MORPH_CLOSE, knot_kernel, iterations=3)
    knot_img = cv2.morphologyEx(knot_img, cv2.MORPH_DILATE, knot_kernel, iterations=7)
    cv2.imshow('knot', knot_img)


    return holes_img, crack_img, knot_img


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True, help="path to the video file")
    args = vars(ap.parse_args())
    main(args["video"])


        