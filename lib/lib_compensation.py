import numpy as np # type: ignore
from matplotlib import pyplot as plt # type: ignore

def compensate_R(image): 
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

def compensate_RB(image): 
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

def show_compensated_image(originalImage, compensatedImage, title="Compensated Image"):
    # Plotting the compensated image
    plt.figure(figsize = (20, 20))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(originalImage)
    plt.subplot(1, 2, 2)
    plt.title("RB Compensated Image")
    plt.imshow(compensatedImage)
    plt.show()