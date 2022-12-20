import cv2

import config
from src.scan_img import scan_img
from image_slicer import slice
from src.utils import check_slice, find_movement
import time
import os

video_name = "img-6.mp4"
cap = cv2.VideoCapture(f"{video_name}")
# cap = cv2.VideoCapture(0)

object_detector = cv2.createBackgroundSubtractorMOG2()
movements = []

saved_img_path = f"{config.TEMP_DIR}/{config.TEMP_IMG}"
if not os.path.exists(saved_img_path):
    os.mkdir(saved_img_path)


while True:
    ret, frame = cap.read()
    if ret:
        H, W = frame.shape[0], frame.shape[1]
        # cut the edges of the video 
        frame = frame[config.H_CUT_SIZE:H-config.H_CUT_SIZE, config.W_CUT_SIZE:W-config.W_CUT_SIZE]

        scanned_img, frame = scan_img(frame, save_steps=False, show_steps=False)

        if scanned_img is not None:
            scanned_img = cv2.resize(scanned_img, (config.H, config.W))
            cv2.imwrite(saved_img_path, scanned_img)		
            slice(saved_img_path, 9)

            slices = {
                "p1": check_slice(cv2.imread(f"{config.TEMP_DIR}/frame_01_01.png")),
                "p2": check_slice(cv2.imread(f"{config.TEMP_DIR}/frame_01_02.png")),
                "p3": check_slice(cv2.imread(f"{config.TEMP_DIR}/frame_01_03.png")),
                "p4": check_slice(cv2.imread(f"{config.TEMP_DIR}/frame_02_01.png")),
                "p5": check_slice(cv2.imread(f"{config.TEMP_DIR}/frame_02_02.png")),
                "p6": check_slice(cv2.imread(f"{config.TEMP_DIR}/frame_02_03.png")),
                "p7": check_slice(cv2.imread(f"{config.TEMP_DIR}/frame_03_01.png")),
                "p8": check_slice(cv2.imread(f"{config.TEMP_DIR}/frame_03_02.png")),
                "p9": check_slice(cv2.imread(f"{config.TEMP_DIR}/frame_03_03.png"))
            }

            movement = find_movement(slices)
            print(movement)
            if movement == "none":
                continue
            elif movements.count(movement) > 3:
                with open(f"{config.MOV_FILE}", "w") as f:
                    f.write(movement)
                time.sleep(40)
            movements.append(movement)
            movements = movements[-50:]
        else:
            print("Table not found!")
        
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(30)
        if key == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
print(movements)