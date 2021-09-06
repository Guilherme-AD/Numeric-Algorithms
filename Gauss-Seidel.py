#---------------------------------------------#
#  Gauss-Seidel Linear System Numeric Solver  #
#                  09/05/21                   #
#---------------------------------------------#

import numpy as np

#Setup 
n = 3
aMat = np.array([[4,-2,-1],[-5,-15,-9],[2,2,-5]])
xMat = np.zeros(n)
xMat_prev = np.zeros(n)
bMat = np.array([1,-9,-8])
betaMat = np.zeros(n)
errorVal = 1e-3
errorCalc = 0.0
itNum = 0
itMax = 1000

#Matrix check
for i in range(0, n):
    if aMat[i, i] == 0:
        print("Matrix has at least one element in main diagonal which equals 0")

#Sassenfeld Criteria
for i in range(0, n):
    for j in range(0 , i):
        betaMat[i] = betaMat[i] + np.abs(aMat[i, j])*betaMat[j]
    for j in range(1+i, n):
        betaMat[i] = betaMat[i] + np.abs(aMat[i, j])
    betaMat[i] = betaMat[i] / np.abs(aMat[i, i])
if np.max(betaMat) < 1.:
    print("Sassenfeld Criteria fulfilled. Method will converge. Max beta = " + str(np.max(betaMat)))
    multiplier = (np.max(betaMat)/ 1 + np.max(betaMat))
else:
    print("Sassenfeld Criteria not fulfilled. Cannot guarantee convergence.")
    multiplier = 1000
    
#Iterative process
for k in range(0, itMax):
    print(str(xMat[:]))
    for i in range(0, n):
        xMat[i] = bMat[i]
        for j in range(0, i):
            xMat[i] = xMat[i] - aMat[i, j]*xMat[j]
        for j in range(i+1, n):
            xMat[i] = xMat[i] - aMat[i, j]*xMat_prev[j]
        xMat[i] = xMat[i] / aMat[i, i] 
    #Error calculation
    errorCalc = multiplier * np.max(np.abs(xMat - xMat_prev))
    itNum += 1
    print("Iteration nº" + str(itNum) + " // Current errorValue = " + str(errorCalc))
    if(errorCalc < errorVal):
        break
    xMat_prev = np.copy(xMat)

#Final verification
finalVal = np.matmul(aMat, xMat)
print("Final results are: " + str(xMat[:]))
print("With residue: " + str(np.max(np.abs(bMat - finalVal))))
