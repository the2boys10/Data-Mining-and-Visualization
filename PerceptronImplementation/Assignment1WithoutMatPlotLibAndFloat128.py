#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Main method runs all tests including binary training, binary testing, multi-class training, multi-class testing
def main():
    # ShuffleVal, if this is set to 1 will shuffle data else will leave data unshuffled.
    shuffleVal = 1
    # howManyTests, how many times to train networks from base values, will only be different for each network if shuffleval is 1.
    howManyTests = 200
    # iterations, how many times to repeatidly pass our data through our testing method.
    iterations = 20
    # get the trainign data and store it in an array.
    trainingData = readFile("train.data")
    # get the trainign data and store it in an array. 
    testingData = readFile("test.data")
    testAndTrainBinary(iterations, trainingData, testingData, shuffleVal, howManyTests)
    print("\n\n--------------------------------------------------------------------------\n--------------------------------------------------------------------------\n")
    print("Classification 1 vs all")
    print("\n\n--------------------------------------------------------------------------\n--------------------------------------------------------------------------\n")
    # for each coefficient we would like to test
    for coef in ([0, 0.01, 0.1, 1, 10, 100]):#, 0.01, 0.1, 1, 10, 100
        testAndTrainMultiClass(coef, iterations, trainingData, testingData, shuffleVal, howManyTests)
    return

def testAndTrainMultiClass(coef, iterations, trainingData, testingData, shuffleVal, howManyTests):
    # initialise all our variables to store the accuracy of our multi-class network
    pred1Act1 = 0
    pred2Act1 = 0
    pred3Act1 = 0
    pred1Act2 = 0
    pred2Act2 = 0
    pred3Act2 = 0
    pred1Act3 = 0
    pred2Act3 = 0
    pred3Act3 = 0
    missclassificationsMulti1 = np.zeros(iterations)
    missclassificationsMulti2 = np.zeros(iterations)
    missclassificationsMulti3 = np.zeros(iterations)
    # for how many tests we would like to run
    for tests in range(0,howManyTests):
        # create a new multi-class network to classify 1 or not 1
        one = perceptTrainOneVsAll(coef, iterations, trainingData, 1, shuffleVal, missclassificationsMulti1)
        # create a new multi-class network to classify 2 or not 2
        two = perceptTrainOneVsAll(coef, iterations, trainingData, 2, shuffleVal, missclassificationsMulti2)
        # create a new multi-class network to classify 3 or not 3
        three = perceptTrainOneVsAll(coef, iterations, trainingData, 3, shuffleVal, missclassificationsMulti3)
        # for each record in our test data
        for features, actualClass in testingData:
            # we take the arg-max of each of our multi-class networks which
            # is the class we predicted.
            valueFound = np.argmax([perceptTest(one,features),perceptTest(two,features),perceptTest(three,features)])+1
            # if the actual class is 1
            if(actualClass == 1):
                # and we predicted 1 then we predicted correctly
                if(valueFound == 1):
                    pred1Act1 += 1
                # else we predicted class 1 when it was class 2
                elif(valueFound == 2):
                    pred2Act1 += 1
                # else we predicted class 1 when it was class 3
                else:
                    pred3Act1 += 1
            # if the actual class is 2
            if(actualClass == 2):
                # and we predicted 2 then we predicted correctly
                if(valueFound == 2):
                    pred2Act2 += 1
                # else we predicted class 2 when it was class 1
                elif(valueFound == 1):
                    pred1Act2 += 1
                # else we predicted class 2 when it was class 3
                else:
                    pred3Act2 += 1
            # if the actual class is 3
            if(actualClass == 3):
                # and we predicted 3 then we predicted correctly
                if(valueFound == 3):
                    pred3Act3 += 1
                # else we predicted class 3 when it was class 1
                elif(valueFound == 1):
                    pred1Act3 += 1
                # else we predicted class 3 when it was class 2
                else:
                    pred2Act3 += 1
    print("\nResults found for coef %0.2f" % coef)
    # prints out the accuracy as well as the table representing the results of our testing.
    print("\t\t\tActually 1\tActually 2\tActually 3\n\tPredicted 1\t"+str(pred1Act1) + "\t\t" + str(pred1Act2) + "\t\t" +str(pred1Act3))
    print("\tPredicted 2\t"+str(pred2Act1) + "\t\t" + str(pred2Act2)+ "\t\t" +str(pred2Act3))
    print("\tPredicted 3\t"+str(pred3Act1) + "\t\t" + str(pred3Act2) + "\t\t" +str(pred3Act3)+"\n\n")
    print("\tOverall accuracy of predictions\t"+ str((pred1Act1+pred2Act2+pred3Act3)/(pred2Act1+pred3Act1+pred1Act2+pred1Act3+pred1Act1+pred2Act2+pred2Act3+pred3Act2+pred3Act3)))
    print("\tAccuracy of predicting class 1\t"+ str((pred1Act1+pred2Act2+pred2Act3+pred3Act3+pred3Act2)/(pred2Act1+pred3Act1+pred1Act2+pred1Act3+pred1Act1+pred2Act2+pred2Act3+pred3Act2+pred3Act3)))
    print("\tAccuracy of predicting class 2\t"+ str((pred2Act2+pred1Act1+pred1Act3+pred3Act1+pred3Act3)/(pred2Act1+pred3Act1+pred1Act2+pred1Act3+pred1Act1+pred2Act2+pred2Act3+pred3Act2+pred3Act3)))
    print("\tAccuracy of predicting class 3\t"+ str((pred3Act3+pred1Act1+pred1Act2+pred2Act1+pred2Act2)/(pred2Act1+pred3Act1+pred1Act2+pred1Act3+pred1Act1+pred2Act2+pred2Act3+pred3Act2+pred3Act3)))
    # print out the individual networks weights
    print("\tWeights of 1 vs all = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f" % (one[0][0], one[0][1], one[0][2], one[0][3], one[1]))
    print("\tWeights of 2 vs all = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f" % (two[0][0], two[0][1], two[0][2], two[0][3], two[1]))
    print("\tWeights of 3 vs all = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f" % (three[0][0], three[0][1], three[0][2], three[0][3], three[1]))
    missclassificationsMulti1 = missclassificationsMulti1/(howManyTests)
    missclassificationsMulti2 = missclassificationsMulti2/(howManyTests)
    missclassificationsMulti3 = missclassificationsMulti3/(howManyTests)
    print("Accuracy during training class 1 vs all")
    draw_graph(missclassificationsMulti1,iterations,"1 vs all coef("+str(coef)+")")
    print("Accuracy during training class 2 vs all")
    draw_graph(missclassificationsMulti2,iterations,"2 vs all coef("+str(coef)+")")
    print("Accuracy during training class 3 vs all")
    draw_graph(missclassificationsMulti3,iterations,"3 vs all coef("+str(coef)+")")
    print("\n\n--------------------------------------------------------------------------\n--------------------------------------------------------------------------\n")
    


def testAndTrainBinary(iterations, trainingData, testingData, shuffleVal, howManyTests):
    print("\n\n--------------------------------------------------------------------------\n--------------------------------------------------------------------------\n")
    print("Binary classifier")
    print("\n\n--------------------------------------------------------------------------\n--------------------------------------------------------------------------\n")
    #Declare variables to store the results of all 3 binary networks.
    found1In1vs2P = 0
    found1In1vs2FP = 0
    found2In1vs2P = 0
    found2In1vs2FP = 0
    found2In2vs3P = 0
    found2In2vs3FP = 0
    found3In2vs3P = 0
    found3In2vs3FP = 0
    found3In3vs1P = 0
    found3In3vs1FP = 0
    found1In3vs1P = 0
    found1In3vs1FP = 0
    missclassifications1 = np.zeros(iterations)
    missclassifications2 = np.zeros(iterations)
    missclassifications3 = np.zeros(iterations)
    # for the amount of tests that we would like to perform
    for tests in range(0,howManyTests):
        # create a binary perceptron to compare class 1 and class 2.
        oneVsTwo = perceptTrainOneVsOne(iterations, trainingData, 1, 2, shuffleVal, missclassifications1)
        # create a binary perceptron to compare class 2 and class 3.
        twoVsThree = perceptTrainOneVsOne(iterations, trainingData, 2, 3, shuffleVal, missclassifications2)
        # create a binary perceptron to compare class 1 and class 3.
        oneVsThree = perceptTrainOneVsOne(iterations, trainingData, 1, 3, shuffleVal, missclassifications3)
        # run each record of our testing data through our test method
        for features, actualClass in testingData:
            #If the record we are processing is either class 1 or class 2
            if(actualClass == 1 or actualClass == 2):
                # test the network if the activation is above 0 then the record 
                # is what the network was trained to find in this case class 1
                if(perceptTest(oneVsTwo, features)>0):
                    # if the actual class was 1 then we predicted correctly
                    if(actualClass == 1):
                        found1In1vs2P += 1
                    # else we predicted falsely
                    else:
                        found1In1vs2FP += 1
                # else its the opposite class (class 2)
                else:
                    # if it was the other class we predicted correctly
                    if(actualClass == 2):
                        found2In1vs2P += 1
                    # else we predicted falsely
                    else:
                        found2In1vs2FP += 1
            #If the record we are processing is either class 2 or class 3
            if(actualClass == 2 or actualClass == 3):
                # test the network if the activation is above 0 then the record 
                # is what the network was trained to find in this case class 2
                if(perceptTest(twoVsThree, features)>0):
                    # if the actual class was 2 then we predicted correctly
                    if(actualClass == 2):
                        found2In2vs3P += 1
                    # else we predicted falsely
                    else:
                        found2In2vs3FP += 1
                # else if the actual class was 3 then we predicted correctly
                else:
                    if(actualClass == 3):
                        found3In2vs3P += 1
                    # else we predicted falsely
                    else:
                        found3In2vs3FP += 1
            # if the class we are testing is either 3 or 1
            if(actualClass == 3 or actualClass == 1):
                # test the network if the activation is above 0 then the record 
                # is what the network was trained to find in this case class 1
                if(perceptTest(oneVsThree, features)>0):
                    # if the class was actually 1 then we predicted correctly
                    if(actualClass == 1):
                        found1In3vs1P += 1
                    # else we predicted falsely
                    else:
                        found1In3vs1FP += 1
                # else we predicted 3 and it was actually 3 then we predicted
                # correctly
                else:
                    if(actualClass == 3):
                        found3In3vs1P += 1
                    # else we predicted falsely.
                    else:
                        found3In3vs1FP += 1 
    # print the network tables for network 1 vs 2.
    printNetworkResults(found1In1vs2P,found1In1vs2FP,found2In1vs2FP, found2In1vs2P,"1","2")
    print()
    # print the network weights for network 1 vs 2
    print("\tWeights of 1 vs 2 = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f" % (oneVsTwo[0][0], oneVsTwo[0][1],oneVsTwo[0][2],oneVsTwo[0][3],oneVsTwo[1]))
    # print the network tables for network 2 vs 3.
    printNetworkResults(found2In2vs3P,found2In2vs3FP,found3In2vs3FP,found3In2vs3P,"2","3")
    print()
    # print the network weights for network 2 vs 3
    print("\tWeights of 2 vs 3 = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f" %(twoVsThree[0][0], twoVsThree[0][1],twoVsThree[0][2],twoVsThree[0][3],twoVsThree[1]))
    # print the network tables for network 1 vs 3.
    printNetworkResults(found3In3vs1P,found3In3vs1FP,found1In3vs1FP,found1In3vs1P,"3","1")
    print()
    # print the network weights for network 1 vs 3
    print("\tWeights of 1 vs 3 = [%0.2f, %0.2f, %.2f, %.2f]    Bias = %.0f\n\n" %(oneVsThree[0][0], oneVsThree[0][1],oneVsThree[0][2],oneVsThree[0][3],oneVsThree[1]))
    missclassifications1 = missclassifications1/(howManyTests)
    missclassifications2 = missclassifications2/(howManyTests)
    missclassifications3 = missclassifications3/(howManyTests)
    print("Accuracy during training class 1 vs 2")
    draw_graph(missclassifications1,iterations,"1 vs 2")
    print("Accuracy during training class 2 vs 3")
    draw_graph(missclassifications2,iterations,"2 vs 3")
    print("Accuracy during training class 3 vs 1")
    draw_graph(missclassifications3,iterations,"3 vs 1")
    
    return


# read file x and output an array containing a numpy array of the features and the class of the record
def readFile(x):
    # open the file
    file_object  = open(x, "r")
    # get all the lines
    y = file_object.readlines()
    # close the file
    file_object.close()
    b = []
    # for every record
    for val in y:
        # split it on ","
        a = val.split(",")
        # get the class of the record.
        a[4] = a[4][6:7]
        c = []
        # add all features to an empty array
        for val2 in range(0,len(a)-1):
            c.append(float(a[val2]))
        # convert the array to a numpy array
        c = np.array(c,dtype=np.float)
        # put the numpy array and the class together inside a new array
        # add it to an array of records.
        temp = [c,int(a[4])]
        b.append(temp)
        # return the array
    return b

# method to train 1 vs all networks takes Perams:
    #regCoef the value of the regularisation coeficient
    #maxIter the amount of iterations to push our data through our network
    #Data the array of all data,
    #ClassValue, the classvalue the network is setup to identify
    #shuffleVal, if we would like to shuffle the data
def perceptTrainOneVsAll(regCoef, maxIter, Data, ClassValue, shuffleVal, missClassificationsMade):
    #create a numpy array made of 4 0's
    weights = np.array([0,0,0,0],dtype=np.float)
    #initialise bias to 0
    bias = 0
    # run the system for maxIter times
    for i in range(0, maxIter):
        mistakesMadeInCurrentIteration = 0
        howManyTested = 0
        # if shuffleval is 1 then shuffle the data
        if shuffleVal == 1: shuffle(Data)
        # for each record split on features and class
        for features, classEx in Data:
            howManyTested +=1
            # if its the class we are looking for set the temp class to be 1
            # else set it to be 0
            tempSetNum = 1 if classEx == ClassValue else -1
            # perform matrix multiplication on features and weights, add the 
            # bias (activation term)
            a = np.dot(weights,features) + bias
            # if the network activated and it shouldn't have
            if(tempSetNum*a<=0):
                # update the weights and bias.
                weights = np.subtract(np.add(weights,np.multiply(tempSetNum,features)),np.multiply(2*regCoef,(weights)))
                bias += tempSetNum
                mistakesMadeInCurrentIteration += 1
        missClassificationsMade[i] += (howManyTested-mistakesMadeInCurrentIteration)/howManyTested
    # return the weights and bias.
    return weights,bias

# method to train binary network, takes Perams:
    #maxIter the amount of iterations to push our data through our network
    #Data the array of all data,
    #ClassValue, the classvalue the network is setup to identify
    #ClassValue2, the classvalue the network is setup to identify distinguish against
    #shuffleVal, if we would like to shuffle the data
def perceptTrainOneVsOne(maxIter, Data, ClassValue, ClassValue2, shuffleVal, missClassificationsMade):
    #create a numpy array made of 4 0's
    weights = np.array([0,0,0,0],dtype=np.float)
    #initialise bias to 0
    bias = 0
    # run system for maxIter times
    for i in range(0, maxIter):
        mistakesMadeInCurrentIteration = 0
        howManyTested = 0
        # if shuffleval is 1 then shuffle the data
        if shuffleVal == 1: shuffle(Data)
        # for each record split on features and class
        for features, classEx in Data:
            #if the records class is either of the two class's we would like to
            #distinguish
            if(classEx == ClassValue or classEx == ClassValue2):
                howManyTested += 1
                # set the tempclass to be 1 if it is the first class
                # else set it to -1.
                tempSetNum = 1 if classEx==ClassValue else -1
                # perform matrix multiplication on features and weights, add the 
                # bias (activation term)                
                a = np.dot(weights,features) + bias
                # if the network activated and it shouldn't have
                if(tempSetNum*a<=0):
                    # update the weights and bias.
                    weights = np.add(weights,np.multiply(tempSetNum,features))
                    bias += tempSetNum
                    mistakesMadeInCurrentIteration += 1
    # return the weights and bias
        missClassificationsMade[i] += (howManyTested-mistakesMadeInCurrentIteration)/howManyTested
    return weights,bias

# method to test the neural network takes the network and record.
def perceptTest(network,record):
    # return the activation value.
    return np.dot(network[0],record) + network[1]

# print the binary neural network results, takes the Perams:
    # TP, True Positives
    # FP, False Positives
    # FN, False Negatives
    # TN, True Negatives
    # index1, class 1
    # index2, class 2
def printNetworkResults(TP, FP, FN, TN, index1, index2):
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    precision = 0
    if(TP+FP!=0):
        precision = (TP)/(TP+FP)
    recall = 0
    if(TP+FN!=0):
        recall = (TP)/(TP+FN)
    falsePositiveRate = 0
    if(FP+TN!=0):
        falsePositiveRate = (FP)/(FP+TN)
    fscore = 0
    if(precision+recall!=0):
        fscore = (2*precision*recall)/(precision+recall)
    print("Classification " + index1+" vs "+index2+"\n"+
          "\n\t\t\tActually "+index1+"\tActually "+index2+"\n"+
          "\tPredicted "+index1+"\t"+str(TP) + "\t\t" + str(FP) + 
          "\n\tPredicted "+index2+"\t" +str(FN) + "\t\t" + str(TN)+"\n\n"+
          "\tAccuracy\t\t"+str(accuracy)+"\n"+
          "\tPrecision\t\t"+str(precision)+"\n"+
          "\tRecall\t\t\t"+str(recall)+"\n"+
          "\tfalsePositiveRate\t"+str(falsePositiveRate)+"\n"+
          "\tfscore\t\t\t"+str(fscore)+"\n")
    return

def draw_graph(data, episode_count, title):
    print("no matplotlib")
    return

# import numpy and import shuffle and call the main method.
import numpy as np
from random import shuffle
with np.errstate(over='ignore'):
    main()