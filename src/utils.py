import cv2
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, "../")
import config


def find_movement(slices):
	values = list(slices.values())
	if values.count("current") > 1 or values.count("target") > 1:
		return "none"
	from_ = ""
	to_ = ""
	for key, value in slices.items():
		if value == "current":
			from_ = key
		elif value == "target":
			to_ = key
	
	if from_ != "" and to_ != "":
		return from_ + to_
	else:
		return "none"
	

def check_slice(slice):
	if has_triangle(slice):
		return "target"
	elif has_circle(slice):
		return "current"
	else:
		return "none"


def preprocess(img, test=False):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ditaled_img = cv2.dilate(gray, (13, 13), iterations=5)
	blurred_img = cv2.GaussianBlur(ditaled_img, (3, 3),0)
	edged_img = cv2.Canny(blurred_img, 30, 50)
	ditaled_img_2 = cv2.dilate(edged_img, (13, 13), iterations=5) 
	final_img = ditaled_img_2

	if test:
		cv2.imshow("ditaled_img", ditaled_img)
		cv2.imshow("blurred_img", blurred_img)
		cv2.imshow("edged_img", edged_img)
		cv2.imshow("ditaled_img_2", ditaled_img_2)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		# pass

	return final_img

def has_triangle(img, test=False):
	edged_img = preprocess(img, test)
	contours, hierarchy = cv2.findContours(edged_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	has = False
	for cnt in contours:
		approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
		if len(approx) == 3 and cv2.contourArea(cnt) > 10:
			has = True
			break
	if test:
		print(f"Contours number: {len(contours)}")
		if has:
			img = cv2.drawContours(img, [cnt], -1, (0,255,0), 2)
		else:
			print("Triangle not found")
		cv2.imshow("Shapes", img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		return has

def has_circle(img, test=False):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, config.minDist, param1=config.param1, param2=config.param2, minRadius=config.minRadius, maxRadius=config.maxRadius)
	has = False
	if circles is not None and len(circles) > 0:
		has = True

	if test:
		if circles is not None:
			circles = np.uint16(np.around(circles))
			for i in circles[0,:]:
				cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
		cv2.imshow('img', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		return has


def resize_img(img):
    max_dim = max(img.shape)
    if max_dim > config.DIM_LIMIT:
        resize_scale = config.DIM_LIMIT / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
    return img


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect


def four_point_transform(img, rect):
	(tl, tr, br, bl) = rect
	
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

	return warped


def pil_to_cv2_image(pil_img):
	img_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) 
	return img_array


def cv2_to_pil_image(img):
	pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	return pil_img


if __name__ == "__main__":

	img = cv2.imread('test_2.png')
	print(has_circle(img))
	print(has_triangle(img))
	# print(has_triangle(img, test=True))

	img = cv2.imread('test_1.png')
	print(has_circle(img, test=True))
	# print(has_triangle(img, test=True))
	print(has_circle(img))
	print(has_triangle(img))
	
	img = cv2.imread('test_3.png')
	print(has_circle(img))
	print(has_triangle(img))