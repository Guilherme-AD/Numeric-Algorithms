#--------#--------#--------#--------#--------#--------#--------#--------#--------#--------#
#   Householder algorithm for transforsming diagonal matrices into tridiagonal matrices   #
#                                       20/07/21                                          #
#--------#--------#--------#--------#--------#--------#--------#--------#--------#--------#
import numpy as np
import math
from QR-Algorithm import QRAlgorithm 

#Householder transform function
def HH_Transform(A):
    shape = A.shape[0]
    shape_aux = shape
    I = np.identity(shape)
    HT = I 
    for i in range(0,shape-2):
        #Checks the signal of the first element from the n-i dimensional matrix
        if A[i,(i+1)] != 0:
            sign = np.sign(A[i,(i+1)])
        else:
            sign = 1
        #E vector
        e = np.zeros((shape_aux-1,1))
        e[0,0] = 1
        #Obtaining w1_ vector
        w_sub = (A[(i+1):,i]).reshape(shape_aux-1,1) + sign*np.linalg.norm(A[(i+1):,i])*e 
        w_sub = np.squeeze(w_sub)
        #Left-side matrix multiplication
        for j in range(i, shape): 
            Hw = A[(i+1):,j]-(2*((np.dot(A[(i+1):,j],w_sub))/(np.dot(w_sub,w_sub)))*w_sub)
            A[(i+1):,j] = Hw
            if j == i: #Symmetry
                A[i,(j+1):] = Hw
        #Obtaining Ht matrix to find eigenvalues 
            if i == 0:
                Ht = I[j,(i+1):]-(2*((np.dot(I[j,(i+1):],w_sub))/(np.dot(w_sub,w_sub)))*w_sub)
                HT[j,(i+1):] = Ht
            else:
                Ht = HT[j,(i+1):]-(2*((np.dot(HT[j,(i+1):],w_sub))/(np.dot(w_sub,w_sub)))*w_sub)
                HT[j,(i+1):] = Ht
        #Right-side matrix multiplication
        for j in range(i+1, shape): 
            Hw = A[j,(i+1):]-(2*((np.dot(A[j,(i+1):],w_sub))/(np.dot(w_sub,w_sub)))*w_sub)
            A[j,(i+1):] = Hw
        shape_aux -= 1
    resVector = [A, HT]
    return resVector
    
#--------#--------#Eigenvetors & Eigenvalues test--------#--------# 

A_test1 = np.array([[2.,4.,1.,1.],[4.,2.,1.,1.],[1.,1.,1.,2.],[1.,1.,2.,1.]]) #Diagonal test matrix
HH_Results = HH_Transform(A_test1)
autoValVec = QRAlgorithm(HH_Results[0], True, HH_Results[1]) #Applying QR factorization
print("Resulting Tridiagonal Matrix: \n")
print(HH_Results[0])
for i in range(0, A_test1.shape[0]):
            print("Eigenvalue " + str(i+1) + ": "+ str(autoValVec[0][i,i])) 
            print("#---------#---------#")
print("Eigenvector matrix: \n")
print(str(autoValVec[1]) + '\n') #Normalized


