import cv2 
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import numpy as np
import pytesseract
from scipy.ndimage import interpolation as inter

#Image Preprocessing Techniques

def get_grayscale(image):
    image=cv2.imread(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def remove_noise(image):
    return cv2.medianBlur(image,5)
 
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def canny(image):
    return cv2.Canny(image, 100, 200)

#Skew corrections in case of inputs of different orientations/angles

def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(image, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
          borderMode=cv2.BORDER_REPLICATE)

    return rotated

# Method to Getting Output

def output(file_path):
    gray=get_grayscale(file_path)
    opening_img = opening(gray)
    thresh=thresholding(gray)
    canny_img=canny(gray)
    rotated=correct_skew(thresh)

    #We can use the best applicable pre-processing technique based on our problem statement. I have tried to create methods for all major ones.
    #cv2.imshow("gray",gray)
    #cv2.imshow("thresholded",thresh)
    #cv2.imshow("canny",canny_img)
    #cv2.imshow("opening",opening_img)
    #cv2.waitKey(0)

    #Setting pytesseract configuration 
    custom_config = r'--oem 3 --psm 6'
    pytesseract.pytesseract.tesseract_cmd =r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

    #Extracting Text From File
    scanned_text=pytesseract.image_to_string(rotated, config=custom_config)

    #Saving Extracted Text to a Text File
    text_file = open("G:/scanned_data.txt", "w")
    text_file.write(scanned_text)
    text_file.close()


#To upload image file. Can be customized to take live input using OpenCV.
root = tk.Tk() 
root.withdraw() 
file_path = filedialog.askopenfilename() 
output(file_path)





