#a grid search function for the BERT hyperparameters recommended in the original paper
#outputs results to a text file at location resultsFileName
def bertGridSearch(trainDataset, testDataset, resultsFileName, BERTType, epochs=[2,3,4], batchSizes=[16.32], learningRates=[5e-5, 3e-5, 2e-5]):
    import fineTunedBert
    for epoch in epochs:
        for batchSize in batchSizes:
            for learningRate in learningRates:
                metrics = fineTunedBert.bertRunner(trainDataset, testDataset, batchSize, 32, BERTType, fineTunedBert.MAP_NAME_TO_HANDLE, fineTunedBert.MAP_MODEL_TO_PREPROCESS, epoch, learningRate)
                with open(resultsFileName, 'a') as f:
                    f.write('%s/%s/%s: \t accuracy: %s, \t precision: %s, \t recall: %s \n' % (epoch, batchSize, learningRate, metrics[0], metrics[1], metrics[2]))

def confirmTrainTestSplit(trainPath, testPath):
    import os
    trainTargetPath = os.path.join(trainPath, 'target')
    trainNonTargetpath = os.path.join(trainPath, 'non_target')
    testTargetpath = os.path.join(testPath, 'target')
    testNonTargetPath = os.path.join(testPath, 'non_target')
    trainList = os.listdir(trainTargetPath) + os.listdir(trainNonTargetpath)
    testList = os.listdir(testTargetpath) + os.listdir(testNonTargetPath)
    return [username for username in trainList if username in testList]

#deletes user entries that appear in both the test and train set if there are any
def deleteSharedEntries():
    import os
    for fileName in confirmTrainTestSplit('./train_datasets/sizelessCleanedSet/textual/','./test_datasets/mediumCleanedSet/textual/'):
        if(os.path.exists(os.path.join('test_datasets', 'sizelessDependentSet','textual', 'target', fileName))):
            os.remove(os.path.join('test_datasets', 'sizelessDependentSet','textual', 'target', fileName))
        if(os.path.exists(os.path.join('test_datasets', 'sizelessDependentSet','textual', 'non_target', fileName))):
            os.remove(os.path.join('test_datasets', 'sizelessDependentSet','textual', 'non_target', fileName))

def splitToCommentSizes(datasetName, trainFlag, delimiter, sizes):
    import os
    
    if(trainFlag):
        datasetHome = 'train_datasets'

    else:
        datasetHome = 'test_datasets'

    targetpath = os.path.join(datasetHome, datasetName, 'textual', 'target')
    nonTargetPath = os.path.join(datasetHome, datasetName, 'textual', 'non_target')

    targetCommentsAsString = []
    targetCommentAuthors = []
    nonTargetCommentsAsString = []
    nonTargetCommentAuthors = []
    for entry in os.listdir(targetpath):
        with open(os.path.join(targetpath, entry), 'r') as userData:
            targetCommentsAsString.append(userData.read())
            targetCommentAuthors.append(os.path.basename(userData.name))
    for entry in os.listdir(nonTargetPath):
        with open(os.path.join(nonTargetPath, entry), 'r') as userData:
            nonTargetCommentsAsString.append(userData.read())   
            nonTargetCommentAuthors.append(os.path.basename(userData.name)) 
    for size in sizes: 
        newDatasetName = datasetName + str(size)
        newTargetPath = os.path.join(datasetHome, newDatasetName, 'textual', 'target')
        newNonTargetPath = os.path.join(datasetHome, newDatasetName, 'textual','non_target')
        if(not os.path.exists(os.path.join(datasetHome, newDatasetName))):
            os.mkdir(os.path.join(datasetHome, newDatasetName))
            os.mkdir(os.path.join(datasetHome, newDatasetName, 'textual'))
            os.mkdir(newTargetPath)
            os.mkdir(newNonTargetPath) 
        for index in range(len(targetCommentsAsString)):
            with open(os.path.join(newTargetPath, targetCommentAuthors[index]), 'w') as newUserData:
                commentArray = targetCommentsAsString[index].split(delimiter)[:size]
                newUserData.write(delimiter.join(commentArray))
        for index in range(len(nonTargetCommentsAsString)):
            with open(os.path.join(newNonTargetPath, nonTargetCommentAuthors[index]), 'w') as newUserData:
                commentArray = nonTargetCommentsAsString[index].split(delimiter)[:size]
                newUserData.write(delimiter.join(commentArray))

            

def splitToCommentSize(delimiter, targetSize, fileAsString):
    commentArray = fileAsString.split(delimiter)
    return commentArray[:targetSize]