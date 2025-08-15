import numpy as np # type: ignore
from PIL import Image # type: ignore
import cv2 # type: ignore

def cv2_to_pil(image_cv):
    b,g,r = cv2.split(image_cv)
    x,y = b.shape

    array_im = np.zeros((x, y, 3), dtype = "uint8")
    array_im[:, :, 0] = r
    array_im[:, :, 1] = g
    array_im[:, :, 2] = b
    image_pil = Image.fromarray(array_im)
    return image_pil

def check_image_cv(image):
  if(type(image) is np.ndarray):
    image_cv = image
  else:
    image_arr = np.asarray(image)
    b, g, r = cv2.split(image_arr)
    image_cv = cv2.merge([r, g, b])
  return image_cv

def compensate_RB(image): #Pranjali
    # Splitting the image into R, G and B components
    imager, imageg, imageb = image.split()

    # Get maximum and minimum pixel value
    minR, maxR = imager.getextrema()
    minG, maxG = imageg.getextrema()
    minB, maxB = imageb.getextrema()

    # Convert to array
    imageR = np.array(imager,np.float64)
    imageG = np.array(imageg,np.float64)
    imageB = np.array(imageb,np.float64)

    x,y = image.size

    # Normalizing the pixel value to range (0, 1)
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j]=(imageR[i][j]-minR)/(maxR-minR)
            imageG[i][j]=(imageG[i][j]-minG)/(maxG-minG)
            imageB[i][j]=(imageB[i][j]-minB)/(maxB-minB)

    # Getting the mean of each channel
    meanR=np.mean(imageR)
    meanG=np.mean(imageG)
    meanB=np.mean(imageB)


    # Compensate Red and Blue channel
    for i in range(y):
        for j in range(x):
            imageR[i][j]=int((imageR[i][j]+(meanG-meanR)*(1-imageR[i][j])*imageG[i][j])*maxR)
            imageB[i][j]=int((imageB[i][j]+(meanG-meanB)*(1-imageB[i][j])*imageG[i][j])*maxB)

    # Scaling the pixel values back to the original range
    for i in range(0, y):
        for j in range(0, x):
            imageG[i][j]=int(imageG[i][j]*maxG)

    # Create the compensated image
    compensateIm = np.zeros((y, x, 3), dtype = "uint8")
    compensateIm[:, :, 0]= imageB
    compensateIm[:, :, 1]= imageG
    compensateIm[:, :, 2]= imageR

    return compensateIm

def compensate_R(image): #Pranjali
    # Splitting the image into R, G and B components
    imager, imageg, imageb = image.split()

    # Get maximum and minimum pixel value
    minR, maxR = imager.getextrema()
    minG, maxG = imageg.getextrema()
    minB, maxB = imageb.getextrema()

    # Convert to array
    imageR = np.array(imager,np.float64)
    imageG = np.array(imageg,np.float64)
    imageB = np.array(imageb,np.float64)

    x,y = image.size

    # Normalizing the pixel value to range (0, 1)
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j]=(imageR[i][j]-minR)/(maxR-minR)
            imageG[i][j]=(imageG[i][j]-minG)/(maxG-minG)
            imageB[i][j]=(imageB[i][j]-minB)/(maxB-minB)

    # Getting the mean of each channel
    meanR=np.mean(imageR)
    meanG=np.mean(imageG)
    meanB=np.mean(imageB)


    # Compensate Red channel
    for i in range(y):
        for j in range(x):
            imageR[i][j]=int((imageR[i][j]+(meanG-meanR)*(1-imageR[i][j])*imageG[i][j])*maxR)

    # Scaling the pixel values back to the original range
    for i in range(0, y):
        for j in range(0, x):
            imageB[i][j]=int(imageB[i][j]*maxB)
            imageG[i][j]=int(imageG[i][j]*maxG)

    # Create the compensated image
    compensateIm = np.zeros((y, x, 3), dtype = "uint8")
    compensateIm[:, :, 0]= imageB
    compensateIm[:, :, 1]= imageG
    compensateIm[:, :, 2]= imageR

    return compensateIm

def clahe(image):
  image_cv = check_image_cv(image)

  lab = cv2.cvtColor(image_cv, cv2.COLOR_BGR2LAB)
    
  # Split the LAB image into different channels
  l, a, b = cv2.split(lab)
  
  # Apply CLAHE to the L channel
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
  l_clahe = clahe.apply(l)
  
  # Merge the CLAHE enhanced L channel with the a and b channel
  lab_clahe = cv2.merge((l_clahe, a, b))
  
  # Convert the LAB image back to BGR
  enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

  x,y = l.shape
      
  new_image = np.zeros((x, y, 3), dtype = "uint8")
  new_image[:, :, 0] = enhanced_image[:, :, 0]
  new_image[:, :, 1] = enhanced_image[:, :, 1]
  new_image[:, :, 2] = enhanced_image[:, :, 2]
  # result_image = Image.fromarray(new_image)

  return new_image

def gray_world(image):
    # Splitting the image into R, G and B components
    imager, imageg, imageb = image.split()

    # Form a grayscale image
    imagegray=image.convert('L')

    # Convert to array
    imageR = np.array(imager,np.float64)
    imageG = np.array(imageg,np.float64)
    imageB = np.array(imageb,np.float64)
    imageGray=np.array(imagegray, np.float64)

    x,y = image.size

    # Get mean value of pixels
    meanR=np.mean(imageR)
    meanG=np.mean(imageG)
    meanB=np.mean(imageB)
    meanGray=np.mean(imageGray)

    # Gray World Algorithm
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j]=int(imageR[i][j]*meanGray/meanR)
            imageG[i][j]=int(imageG[i][j]*meanGray/meanG)
            imageB[i][j]=int(imageB[i][j]*meanGray/meanB)

    # Create the white balanced image
    whitebalancedIm = np.zeros((y, x, 3), dtype = "uint8")
    whitebalancedIm[:, :, 0]= imageB
    whitebalancedIm[:, :, 1]= imageG
    whitebalancedIm[:, :, 2]= imageR

    return whitebalancedIm

