#--------#--------#--------#--------#--------#--------#--------#--------#--------#--------#
#   QR Algorithm for obtaining eigenvalues and eigenvectors from tridiagonal matrices     #
#                                      20/07/21                                           #    
#--------#--------#--------#--------#--------#--------#--------#--------#--------#--------#

import numpy as np 
import math 
from functools import reduce 

#Generates a rotation matrix
def generateRotationMatrix(shape, c_mat, s_mat, i_mat, j_mat):
    matrixG = np.array(np.zeros((shape,shape)))
    for i in range(0, shape):
        for j in range(0, shape):
            if (i == j) and (i == i_mat or j == j_mat): #Gkk = c for k == i,j
                matrixG[i,j] = c_mat 
            elif (i == j) and (i != i_mat or j != j_mat): #Gkk = 1 for k != i,j
                matrixG[i,j] = 1
    matrixG[i_mat,j_mat] = s_mat
    matrixG[j_mat,i_mat] = -s_mat
    return matrixG

#Triangularization function using rotation matrix
def givensRotation(matrixA):
    QRResults = []
    GMatrices = []
    dimension = matrixA.shape[0]
    matrixAk = matrixA
    for i in range(1, dimension):
        if matrixAk[i,i-1] != 0: 
            if abs(matrixAk[i-1,i-1]) > abs(matrixAk[i,i-1]): #Numerically stable
                tau = (-matrixAk[i,i-1])/matrixAk[i-1,i-1]
                c = 1/(math.sqrt(1+pow(tau,2)))
                s = c*tau
            else:
                tau = (-matrixAk[i-1,i-1])/matrixAk[i,i-1]
                s = 1/(math.sqrt(1+pow(tau,2)))
                c = s*tau
            G = generateRotationMatrix(dimension, c, s, i, i-1)
            GMatrices.append(G)
            matrixAk = np.matmul(G, matrixAk)
    matrixR = matrixAk #Upper triangular
    GMatricesTransp = [np.transpose(GMatrices[i]) for i in range(0, len(GMatrices))] 
    matrixQ = reduce(np.matmul, GMatricesTransp) #np.matmul of all transposed G matrices
    QRResults = [matrixQ, matrixR]
    return QRResults

#QR Algorithm
def QRAlgorithm(A, desloc):
    Ak = A
    shape = A.shape[0] #Matrix A, k = 0
    Vk = np.identity(shape) #Matrix V, k = 0, V0 = I
    errval = 10e-6
    k = 0 #Number of iterations
    for m in range(1, shape): 
        while abs(Ak[m,m-1]) > errval: 
            if k == 0 or desloc == False: 
                mi = 0
            else:
                dk = (Ak[m-1,m-1] - Ak[m,m])/2 #dk = (alpha_n-1 + alpha_n)/2
                if dk >= 0:
                    sign = 1
                else:
                    sign = -1
                mi = (Ak[m,m] + dk - (sign*np.sqrt(pow(dk,2)+pow(Ak[m,m-1],2)))) 
            QRArray = givensRotation((Ak - (np.identity(shape))*mi)) #QR Factorization
            Qk = QRArray[0]
            Rk = QRArray[1]
            Ak = (np.matmul(Rk,Qk) + (np.identity(shape))*mi) #Updates the matrix
            Vk = np.matmul(Vk,Qk) #Updates eigenvectors
            k+=1
    print("->" + str(k) + " Iteracoes do Algoritmo QR")
    results = [Ak,Vk]
    return results 