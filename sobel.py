
from scipy import misc
import sys, getopt
import numpy as np
import math

DETECT_FACTOR = 0.7

# Apply the given mask to the given image. Return an array containing gradient
# magnitude at each calculated point.
def Apply_Gradient_Kernel(imgin,kernel):

    imShape = imgin.shape
    im_xmax = imShape[0]
    im_ymax = imShape[1]
    im_zmax = 1
    if imShape[2]:
        im_zmax = imShape[2]
    

    out_xsize = im_xmax-2
    out_ysize = im_ymax-2
    out_zsize = im_zmax
    outShape = (out_xsize, out_ysize)
    magImg = np.zeros(outShape, 'uint8')

    dxKernel = kernel[0]
    dyKernel = kernel[1]
    dxShape = dxKernel.shape
    dyShape = dyKernel.shape
    
    
    # Step through each pixel in the image and apply the convolution kernels
    aboveCount = 0
    belowCount = 0
    gradM_max = 0
    for i in range(1,im_xmax-2):
        for j in range(1,im_ymax-2):
            for k in range(0,1):
                print(i,j,k)
                M = 0
                dxSum = 0
                dySum = 0

                # Apply dx kernel to estimate dx
                for m in range(0,dxShape[0]):
                    for n in range(0,dxShape[1]):
                        dxSum = dxSum + dxKernel[m,n]*imgin[i-((dxShape[0]-1)/2) + m, j-((dxShape[1]-1)/2) + n, k]
                # Apply dy kernel to estimate dy    
                for m in range(0, dyShape[0]):
                    for n in range(0,dyShape[1]):
                        dySum = dySum + dyKernel[m,n]*imgin[i-((dyShape[0]-1)/2) + m, j-((dyShape[1]-1)/2) + n, k]

                # Calculate the magnitude of the gradient from the partial derivatives
                #print(dxSum)
                #print(dySum)
                
                M = math.sqrt(dxSum**2 + dySum**2)

                if M > gradM_max:
                    gradM_max = M

                
                
                # Put the (un-normalized) magnitude values into the image array
                magImg[i,j] = M

    print aboveCount, belowCount, gradM_max

    # Truncate pixels outside of the grayscale range (0 to 255) to be either
    # 0 or 255
    for i in range(0,magImg.shape[0]):
        for j in range(0, magImg.shape[1]):

            magImg[i,j] = (magImg[i,j]/gradM_max)*255

            #if magImg[i,j] > 255:
                #magImg[i,j] = 255
            #if magImg[i,j] < 0:
                #magImg[i,j] = 0

    return magImg

def Detect_Edges(img):
    imgSize = img.size
    img_xmax = imgSize[0]
    img_ymax = imgSize[1]
    #for i in range(0,img_xmax):
        #for j in range(0, img_ymax):
            



### MAIN ###


# Load the greyscale image in the file given by the first argv into a np array
loadedimg = misc.imread(sys.argv[1])

# Defining the x and y gradient operators for the Sobel Operator.
# The dx approximation mask is the first element of the sobelOp (sobelOp[0])
# dy approximation mask is sobelOp[1]
#           
# sobelOp = [
#            [
#             [-1,0,1],
#             [-2,0,2],  <--- dx partial derivative estimator
#             [-1,0,1],
#            ],
#            [
#             [1,2,1],
#             [0,0,0],    <--- dy partial estimator
#             [-1,-2,-1],
#            ],
#           ]
sobelKernel = np.array([[[-1,0,1],[-2,0,2],[-1,0,1]],[[1,2,1],[0,0,0],[-1,-2,-1]]])

gradientImg = Apply_Gradient_Kernel(loadedimg, sobelKernel)
outfile = sys.argv[2]
gradientName = outfile[0:-4]+"_gradient.jpg"
misc.imsave(gradientName, gradientImg)



def graveyard():
    
    if M > 255:
        M = 255
        aboveCount = aboveCount+1
    if M < 0:
        M = 0
        belowCount = aboveCount+1 


