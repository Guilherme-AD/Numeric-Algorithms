#------------------------------------------------#
# Gauss Elimination Linear System Numeric Solver #
#              MAP3121 - 10/05/21                #
#          Guilherme Alvarenga Dias              #
#------------------------------------------------#

#Auxilary 
import numpy as np
def rowChange(checkMat, line1, line2):
    line1Aux = line1
    while (line1Aux < checkMat.shape[0]) and (checkMat[line1Aux, line2]) == 0:
        line1Aux += 1
    #If entire row is 0s
    if line1Aux == checkMat.shape[0]: 
        return None
    #Swap Row
    checkMat[[line1, line1Aux]] = checkMat[[line1Aux, line1]]
    return checkMat

#Main
def gaussElim(aMat_in, bMat_in):
    import numpy as np
    i = 0 #RowNum
    j = 0 #ColNum
    mult = 0 #Multiplier array
    pivot = 0 #Line pivot

    #Setup
    aMat = aMat_in
    bMat = bMat_in.reshape(aMat.shape[0],1)
    augMat = np.hstack((aMat, bMat))
    m = augMat.shape[0]
    k = augMat.shape[1]
    #Gauss algorithm
    while (i < m) and (j < k):
        #print("Step n°" + str(i+1) + ":" + "\n" + str(augMat))
        #Check consistency
        if np.linalg.det(augMat[:, :-1]) == 0:
            print("Matrix is not consistent / System is undetermined")
            break
        #Search for pivot / Swap rows if pivot == 0 / Pivotal condensation
        if (augMat[i,j] == 0):
            augMat = rowChange(augMat, i, j)
        pivot = augMat[i,j]
        #Define multiplier
        for i_m in range(i+1, m):
            mult = augMat[i_m,j]/pivot
            for j_m in range(j+1, k):
                augMat[i_m,j_m] = augMat[i_m,j_m] - mult*augMat[i,j_m]
            #Store multiplier in empty spaces
            augMat[i_m,j] = mult 
        #Iterate
        i += 1
        j += 1

    #Separating augmented matrix for convenience 
    newBMat = augMat[:, k-1]
    newAMat = augMat[:, 0:k-1]

    #Find results in triangular Matrix
    results = [0]*m
    results[m-1] = (newBMat[m-1]/newAMat[m-1, m-1])
    for i in reversed(range(0, m-1)):
        sumMat = 0
        for j in range(i+1, m):
            sumMat = sumMat + (newAMat[i,j]*results[j])
        results[i] = (newBMat[i] - sumMat)/newAMat[i,i]
    return results

#Implementation Example

#Test3
aMat_in = np.array([[1,0,0,0],[1,0.5,pow(0.5,2),pow(0.5,3)],[1,0.75,pow(0.75,2),pow(0.75,3)],[1,1,1,1]])
bMat_in = np.array([0,0.479,0.682,0.841])
results = gaussElim(aMat_in,bMat_in)
print(results)

#Refinement step
resultsRef = results
refError = 5
while refError > 1e-5:
    rMat =  np.longdouble(np.subtract(bMat_in,np.matmul(aMat_in, resultsRef)))
    corrMat = gaussElim(aMat_in, rMat)
    oldResultsRef = resultsRef
    resultsRef = np.add(resultsRef,corrMat)
    refError = (abs(np.subtract(oldResultsRef,resultsRef))).max()
finalResult = resultsRef
print(finalResult)
