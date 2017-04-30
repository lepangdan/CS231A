# CS231A Homework 0, Problem 2
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
from numpy.linalg import inv
def main():
    # ===== Problem 2a =====
    # Define Matrix M and Vectors a,b,c in Python with NumPy

    M, a, b, c = None, None, None, None

    # BEGIN YOUR CODE HERE
    M=np.array([[1,2,3],[4,5,6],[7,8,9],[0,2,2]])
    a=np.array([[(1)],[(1)],[(0)]])
    b=np.array([[(-1)],[(2)],[(5)]])
    c=np.array([[(0)],[(2)],[(3)],[(2)]])
    print('M',M.shape)
    print('a',a.shape)
    print('b',b.shape)
    print('c',c.shape)
    a1=a.reshape(1,3)
    print('aT',a1)
    aDotb=np.dot(a1,b)
    print('aDotb',aDotb)

    amultiplyb=np.multiply(a,b)
    print('amultiply\n',amultiplyb)

    d=np.multiply(aDotb,M)
    d=np.dot(d,a)
    print('d',d)

    a_repmat=np.matlib .repmat(a1,4,1)
    M=np.multiply(M,a_repmat)
    print('e',M)

    f=np.sort(M,axis=None)
    print('f',f)
    print(f.shape)

    x=np.linspace(1,12,12)
    print(x)
    plt.scatter(x,f)
    plt.show()
    # END YOUR CODE HERE

    # ===== Problem 2b =====
    # Find the dot product of vectors a and b, save the value to aDotb

    aDotb = None

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 2c =====
    # Find the element-wise product of a and b

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 2d =====
    # Find (a^T b)Ma

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 2e =====
    # Without using a loop, multiply each row of M element-wise by a.
    # Hint: The function repmat() may come in handy.

    newM = None

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE

    # ===== Problem 2f =====
    # Without using a loop, sort all of the values 
    # of M in increasing order and plot them.
    # Note we want you to use newM from e.

    # BEGIN YOUR CODE HERE

    # END YOUR CODE HERE


if __name__ == '__main__':
    main()