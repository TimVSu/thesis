import matplotlib.pyplot as plt
import os 


def commentSizePlot(x, accuracies, precisions, recalls):
    x = [0,1,2,5,10, 20, 30, 40, 50, 60, 70, 80, 90]
    #accuracys = [0.5, 0.57, 0.63, 0.69, 0.69, 0.705, 0.72, 0.73, 0.695, 0.69, 0.705, 0.705, 0.715]
    #precisions = [0.0, 0.5975, 0.6477, 0.6969, 0.7564, 0.7892, 0.7468, 0.7893, 0.7519, 0.7037, 0.7252, 0.7556, 0.804]
    #recalls = [0.0, 0.49, 0.5699, 0.6699, 0.5981, 0.6155, 0.6426, 0.6542, 0.6045, 0.6397, 0.6157, 0.6063, 0.6095]

    #accuracys = [0.4838709533214569, 0.5483871102333069, 0.6505376100540161, 0.6720430254936218, 0.6666666865348816, 0.6666666865348816, 0.6666666865348816, 0.6666666865348816, 0.6666666865348816, 0.6666666865348816, 0.6666666865348816, 0.6666666865348816, 0.6666666865348816]
    #precisions = [0.0, 0.563829779624939, 0.6741573214530945, 0.7272727489471436, 0.707317054271698, 0.707317054271698, 0.707317054271698, 0.707317054271698, 0.707317054271698, 0.707317054271698, 0.707317054271698, 0.707317054271698, 0.707317054271698]
    #recalls = [0.0, 0.5520833134651184, 0.625, 0.5833333134651184, 0.6041666865348816, 0.6041666865348816, 0.6041666865348816, 0.6041666865348816, 0.6041666865348816, 0.6041666865348816, 0.6041666865348816, 0.6041666865348816, 0.6041666865348816]


    plt.plot(x, accuracies, label='accuracy')
    #plt.plot(x, f1s, label='f1-score')
    plt.plot(x, precisions, label='precision')
    plt.plot(x, recalls, label='recall')


    plt.xlabel("comments per user")
    plt.ylabel("metrics-score")
    plt.legend()
    plt.savefig('sample_sizes.png')
    plt.show()

def metricsLatextable(sizelessPath, path):
    import seaborn as sns
    lines = []
    with open(sizelessPath, 'r') as gridSearchFile:
        lines = gridSearchFile.readlines()
    accuracySizeless = [line.split('accuracy:')[1][1:7] for line in lines]
    precisionSizeless = [line.split('precision:')[1][1:7] for line in lines]
    recallSizeless = [line.split('recall')[1][1:7] for line in lines]

    with open(path, 'r') as gridSearchFile:
        lines = gridSearchFile.readlines()
        accuracy = [line.split('accuracy:')[1][1:7] for line in lines]
        precision = [line.split('precision:')[1][1:7] for line in lines]
        recall = [line.split('recall')[1][1:7] for line in lines]
    for index in range(0, len(accuracy)): 
        print('&%s&%s&%s&%s&%s&%s& \hline' % (accuracy[index], precision[index], recall[index], accuracySizeless[index], precisionSizeless[index], recallSizeless[index]))


def countCommentAmounts(path, delimiter):
    commentAmounts = [] 
    for entry in os.listdir(path):
        with open(os.path.join(path, entry), 'r') as userData:
            commentAmounts.append(len(userData.read().split(delimiter)))
    return commentAmounts

def getCountOfCommentAmount(commentAmounts, count):
    return sum([entry == count for entry in commentAmounts])

def histPlotComments():
    import pandas as pd 
    import seaborn as sns 
    import matplotlib.pyplot as plt 
        

    allCommentAmountsTargetTrain = [getCountOfCommentAmount(countCommentAmounts('./train_datasets/sizelessCleanedSet/textual/target/', '~\t~'), count ) for count in list(range(300))]
    allCommentAmountsNonTargetTrain = [getCountOfCommentAmount(countCommentAmounts('./train_datasets/sizelessCleanedSet/textual/non_target/', '~\t~'), count ) for count in list(range(300))]

    allCommentAmountsTargetTest = [getCountOfCommentAmount(countCommentAmounts('./test_datasets/sizelessCleanedSet/textual/target/', '~\t~'), count ) for count in list(range(300))]
    allCommentAmountsNonTargetTest = [getCountOfCommentAmount(countCommentAmounts('./test_datasets/sizelessCleanedSet/textual/non_target/', '~\t~'), count ) for count in list(range(300))]

    plt.hist(countCommentAmounts('./test_datasets/sizelessCleanedSet/textual/target/', '~\t~'), bins=25, alpha=0.45, color='red')
    plt.hist(countCommentAmounts('./test_datasets/sizelessCleanedSet/textual/non_target/', '~\t~'), bins=25, alpha=0.45, color='blue')

    print(allCommentAmountsNonTargetTrain)
    print(allCommentAmountsTargetTrain)
    
    plt.title("Comment amount distribution test dataset") 
    
    plt.legend(['ADHD group',  
                'non-ADHD group']) 
    plt.xlabel("comments per user")
    plt.ylabel("number of users")
    plt.savefig("comment_distribution_test.png")
    plt.show()

metricsLatextable('sizelessCleanedSetGridSearch_old.txt', 'mediumCleanedSetGridSearch.txt')