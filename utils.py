import cv2
import numpy as np

def get_brightness(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])

def detect_camera_angle_change(ref_frame, curr_frame, threshold=0.75):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(ref_frame, None)
    kp2, des2 = orb.detectAndCompute(curr_frame, None)

    if des1 is None or des2 is None:
        return True

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return True

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]

    match_ratio = len(good_matches) / max(len(matches), 1)
    return match_ratio < threshold
