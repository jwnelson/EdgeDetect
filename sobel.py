from scipy import misc
import sys, getopt
import numpy as np
import math
import time

DETECT_FACTOR = 0.5

# USAGE:
# python sobel.py infile outfile




# Apply the given kernel to the given image. Return an array containing gradient
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
    outShape = (out_xsize, out_ysize, out_zsize)
    magImg = np.zeros(outShape, 'uint8')

    dxKernel = kernel[0]
    dyKernel = kernel[1]
    dxShape = dxKernel.shape
    dyShape = dyKernel.shape
    
    startTime = time.time()
    # Step through each pixel in the image and apply the convolution kernels
    aboveCount = 0
    belowCount = 0
    gradM_max = 0
    gradM_min = 10000
    for i in range(1,im_xmax-2):
        for j in range(1,im_ymax-2):
            for k in range(0,im_zmax):
                print(i,j,k)
                M = 0
                dxSum = 0
                dySum = 0

                # Apply dx and dy kernels to estimate dx and dy
                for m in range(0,dxShape[0]):
                    for n in range(0,dxShape[1]):
                        dxSum = dxSum + dxKernel[m,n]*imgin[i-((dxShape[0]-1)/2) + m, j-((dxShape[1]-1)/2) + n, k]
                        dySum = dySum + dyKernel[m,n]*imgin[i-((dyShape[0]-1)/2) + m, j-((dyShape[1]-1)/2) + n, k]
                
                        

                # Calculate the magnitude of the gradient from the partial derivatives
                #print(dxSum)
                #print(dySum)
                
                M = math.sqrt(dxSum**2 + dySum**2)

                # Check whether this value is greater than the current greatest value
                # This will be used to normalize the image
                if M > gradM_max:
                    gradM_max = M
                elif M < gradM_min:
                    gradM_min = M

                
                
                # Put the (un-normalized) magnitude values into the image array
                magImg[i,j,k] = M

    print aboveCount, belowCount, gradM_max

    # Normalize pixels to the greyscale range from 0 to 255 using a simple linear scaling function
    for i in range(0,magImg.shape[0]):
        for j in range(0, magImg.shape[1]):

            magImg[i,j] = (magImg[i,j]-gradM_min)*(255/(gradM_max-gradM_min))+0

    endTime = time.time()
    timeElapsed = endTime - startTime
    print("Time Elapsed: ",timeElapsed)
    return magImg

def Simple_Edge(img):
    imgSize = img.shape
    img_xmax = imgSize[0]
    img_ymax = imgSize[1]
    for i in range(0,img_xmax):
        for j in range(0, img_ymax):
            if img[i,j] < DETECT_FACTOR*255:
                img[i,j] = 0
    return img



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

# Defining the Prewitt Operator kernel. Same as Sobel, except all the 2's are
# replaced by 1's so that pixels aren't weighted by their proximity to the center
# pixel
prewittKernel = np.array([[[-1,0,1],[-1,0,1],[-1,0,1]],[[1,1,1],[0,0,0],[-1,-1,-1]]])

# Apply the kernel to the loaded target image. Get a normalized greyscale
# image of gradient magnitudes back
gradientImg = Apply_Gradient_Kernel(loadedimg, sobelKernel)

# Save the gradient image
outfile = sys.argv[2]
gradientName = outfile[0:-4]+"_gradient.jpg"
misc.imsave(gradientName, gradientImg)

# Apply a simple edge detection algorithm to the gradient image
#detectedImg = Simple_Edge(gradientImg)

# Save the simple edge detection image
#edgeName = outfile[0:-4]+"_simpleEdge.jpg"
#misc.imsave(edgeName, detectedImg)


