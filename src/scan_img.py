import cv2
import os
import numpy as np

from config import OUT_DIR
from src.utils import order_points, resize_img, four_point_transform


def get_edges(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    top_hat_img = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)
    black_hat_img = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
    top_black_img = gray_img + top_hat_img - black_hat_img
    filtered_img = cv2.bilateralFilter(top_black_img, 15, 75, 75)
    kernel = np.ones((5,5),np.uint8)
    morph_img = cv2.morphologyEx(filtered_img, cv2.MORPH_CLOSE, kernel, iterations=5)
    blurred_img = cv2.GaussianBlur(filtered_img, (5, 5),0)
    edged_img = cv2.Canny(filtered_img, 30, 50) 
    ditaled_img = cv2.dilate(edged_img, (13, 13), iterations=7)
    ditaled_img_2 = cv2.dilate(ditaled_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    steps = {
        'Gray': gray_img,
        "Top Black": top_black_img,
        "Filtered": filtered_img,
        "Morph": morph_img,
        "Blur": blurred_img,
        "Edged": edged_img,
        "Dilated": ditaled_img,
        "Dilated 2": ditaled_img_2
    }

    return ditaled_img_2, steps


def scan_img(img, img_name="fame", save_steps=True, show_steps=True):
    img = resize_img(img)
    orig_img = img.copy()
    edged_img, img_processing_steps = get_edges(img)

    contours, hierarchy = cv2.findContours(edged_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:5]
    cnt = None
    if contours:
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 100:
                cnt = approx
                pts = cnt.reshape(4, 2)
                rect = order_points(pts)
                scanned_img = four_point_transform(orig_img, rect)
                x, y, w, h = cv2.boundingRect(cnt)
                break

        if cnt is None:
            cnt = max(contours, key = cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)  
            scanned_img = orig_img[y:y+h, x:x+w]        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if save_steps:
            folder = f"{OUT_DIR}/{img_name}"
            if not os.path.exists(folder):
                os.makedirs(folder)

            cv2.imwrite(f"{folder}/1 - Original {img_name}.png", orig_img)
            step = 2
            for proc_name in img_processing_steps.keys():
                cv2.imwrite(f"{folder}/{step} - {proc_name} {img_name}.png", img_processing_steps[proc_name])
                step += 1
            cv2.imwrite(f"{folder}/Document found {img_name}.png", img)

        if show_steps:
            cv2.imshow("Original image", orig_img)
            for proc_name in img_processing_steps.keys():
                cv2.imwrite(f"{proc_name} image.png", img_processing_steps[proc_name])
            cv2.imshow("Document found image", img)
            cv2.imshow("Scanned image", scanned_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return scanned_img, img

    else:

        return None