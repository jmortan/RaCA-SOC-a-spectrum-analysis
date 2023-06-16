import cv2 
import numpy as np 


def extractBoundary(img_name):
    #current extraction from images unable to find holes within image
    #could potentially use contour hierarchies (still need to distinguish between buildings and farm)
    #define within farm object areas of non sampling 
    img_path = "./sample_farm_images/"+img_name
    split = img_name.split(".")
    img_name = split[0]
    file_type = split[1]

    og = cv2.imread(img_path)
    img = cv2.imread(img_path, 0)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = 255 - thresh
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(img)

    denoised = denoiseContours(contours)

    cv2.drawContours(out, denoised, -1, 255, 1)

    cv2.imshow('Original', og) 
    cv2.imshow('Boundary', out)
    result_path = "./sample_farm_images/" + img_name + "_results." + file_type
    print(result_path)
    cv2.imwrite(result_path, out)
    cv2.waitKey(0)

    return denoised 

def denoiseContours(contours): 
    #only want contours that have area above threshold
    #just taking max area for now
    max_area_idx = np.argmax([cv2.contourArea(contour) for contour in contours])
    return contours[max_area_idx]

def constructFarm(img_path): 
    boundary = extractBoundary(img_path)
    return Farm(boundary)

class Farm:
    """
    Represents a Farm
    """
    def __init__(self, boundary, img_path = None):
        #change what the boundary is based on the size
        #it should be possible to scale right?
        #want each point to represent potential sampling place
        #guh how do you even represent this guh
        self.boundary = boundary
        self.img_path = img_path

    def __str__(self):
        return f"{self.boundary})"
