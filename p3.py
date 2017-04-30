# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.



    # BEGIN YOUR CODE HERE
    img1, img2 = None, None
    img1 = misc.imread('./image1.jpg')
    img2 = misc.imread('./image2.jpg')
    #plt.figure(num=1)
    #plt.imshow(img1)
    #plt.figure(num=2)
    #plt.imshow(img2)
    # plt.show()
    # img1_type=type(img1)
    # print(img1_type)
    print(img1.shape)
    print(img1.dtype)
    # END YOUR CODE HERE

    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1.astype(float)
    img11=np.multiply(img1,1.0/256)
    #print(np.max(img11))
    #plt.imshow(img11)
    #plt.show()



    # END YOUR CODE HERE

    # ===== Problem 3c =====
    # Add the images together and re-normalize them 
    # to have minimum value 0 and maximum value 1. 
    # Display this image.

    # BEGIN YOUR CODE HERE
    img111 = np.add(img1, img11)
    img111= np.multiply(img111, 1.0 / 256)
    plt.imshow(img111)
    #plt.show()

    # END YOUR CODE HERE
    '''
    # ===== Problem 3d =====
    # Create a new image such that the left half of 
    # the image is the left half of image1 and the 
    # right half of the image is the right half of image2.


    newImage1 = None

    # BEGIN YOUR CODE HERE
    img1_left=img1[:,:149]
    img2_right=img2[:,150:]
    newImage1=np.concatenate((img1_left ,img2_right),axis=1)
    plt.imshow(newImage1)
    plt.show()
    '''


    # END YOUR CODE HERE

    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd 
    # numbered row is the corresponding row from image1 and the 
    # every even row is the corresponding row from image2. 
    # Hint: Remember that indices start at 0 and not 1 in Python.

    newImage2 = np.random.rand(300,300,3)
    '''
    # BEGIN YOUR CODE HERE
    for idx in range(300):
        if idx%2==0:
            newImage2[idx]=img1[idx]
        else:
            newImage2[idx]=img2[idx]
    plt.imshow(newImage2)
    plt.show()
    '''
    # END YOUR CODE HERE

    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and repmat may be helpful here.


    newImage3 = None

    # BEGIN YOUR CODE HERE
    newImage2 = np.concatenate((img1, img2), axis=1)
    newImage2 = np.reshape(newImage2, (150, 1200, 3))
    #print('jj', newImage2.shape)
    newImage2 = newImage2[:, :600]
    newImage3 = np.reshape(newImage2, (300, 300, 3))
    #plt.imshow(newImage3)
    #plt.show()
    # END YOUR CODE HERE

    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image. 
    # Display the grayscale image with a title.

    # BEGIN YOUR CODE HERE
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    grayimg=rgb2gray(newImage3)
    fig3=plt.figure(num= 3)
    fig3.suptitle('grey image',fontsize=20)

    plt.imshow(grayimg,cmap=plt.get_cmap('gray'))
    plt.show()
    # END YOUR CODE HERE
    plt.close()


if __name__ == '__main__':
    main()