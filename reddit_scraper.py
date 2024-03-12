import praw
import time
import os
from langdetect import detect
from enum import Enum
from langdetect import LangDetectException
import prawcore.exceptions as pe
import redditcleaner
import re

classification = Enum('classification', ['TARGET', 'NON_TARGET', 'UNCLEAR'])
if os.environ.get('CLIENT_ID') is None:
    raise Exception('missing CLIENT_ID environement variable')
if os.environ.get('CLIENT_SECRET') is None:
    raise Exception('missing CLIENT_SECRET environement variable')
if os.environ.get('USER_AGENT') is None:
    raise Exception('missing USER_AGENT environement variable')

CLIENT_ID = os.environ.get('CLIENT_ID')
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')
USER_AGENT = os.environ.get('USER_AGENT')


reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT,
)

def getTargetIds(nameList):
    print('getting target ids')
    return [reddit.subreddit(name).id for name in nameList]

def getSubredditsFromNames(nameList):
    return (reddit.subreddit(name) for name in nameList)

def getSubUsers(targetName, limit):
    iterator = []
    for submission in reddit.subreddit(targetName).new(limit=limit):
        try:
            iterator.append(submission.author)
        except: continue
    print('found users: ', iterator)
    return iterator

def isTargetUser(user, targetIds):

    index = 0
    allcomments = user.comments.hot(limit=350)
    for comment in allcomments:
        print('testing comment# ' + str(index))

        try:
            commentFrom = comment.subreddit_id
            if(commentFrom in targetIds):
                return classification.TARGET
            index+= 1

        except pe.TooManyRequests:
            time.sleep(300)
            index+= 1
            continue

    return classification.NON_TARGET

def userExists(name):
    try:
        reddit.redditor(name).id
    except:
        return False
    return True

#tests wether a specific user already exists in a directory
def userFileAlreadyExists(directory, userName):

    if os.path.exists(os.path.join(directory, 'target') and os.path.exists(os.path.join(directory, 'non_target'))):
        if os.path.exists(os.path.join(directory, 'target', userName + '.txt')) or os.path.exists(os.path.join(directory, 'non_target', userName + '.txt')):
            print(userName + ': file already exists')
            return True
    return False

def textContainsKeywords(text, keywords):
    return (True in [keyword in text for keyword in keywords]) 

def getTextualUserData(user, path, testPath, targetIds, keywords, limit=1000, batchSize=400, fixedBatchSize=True, isTarget=True):
    print('getting user data')
    targetPath = os.path.join(path, 'target')
    nonTargetPath = os.path.join(path, 'non_target')

    try:
        userName = user.name
    except AttributeError:
        return -1
    
    if(userExists(userName) and not userFileAlreadyExists(path, userName) and not userFileAlreadyExists(testPath, userName)):
        print('Working on user: ', userName)
        usableComments = []
        commentsAdded = 0
        for comment in user.comments.new(limit=limit):
            if (commentsAdded >= batchSize):
                break
            try:
                if(hasattr(comment, 'body')):
                    commentBody = redditcleaner.clean(comment.body).encode('ascii', 'ignore').decode('ascii')
                    commentBody = re.sub(r'^https?:\/\/.*[\r\n]*', '', commentBody, flags=re.MULTILINE)
                    commentBody = re.sub(r'^http?:\/\/.*[\r\n]*', '', commentBody, flags=re.MULTILINE)
                    if(detect(commentBody) == 'en'):
                        if(not textContainsKeywords(commentBody, keywords)):
                            print('adding comment# ', str(len(usableComments) + 1), 'for user: ', userName)
                            commentsAdded+= 1
                            usableComments.append(commentBody.lower())
            except LangDetectException:
                print('Language detection exception occured')
                continue
            except pe.TooManyRequests:
                print('too many requests, 5 minute timeout, if the program stops after the timeout wait longer and restart')
                time.sleep(300)
                continue
        if(len(usableComments) >= batchSize or (not fixedBatchSize and len(usableComments) >= 5)):
            if(isTarget):
                saveTextualData(usableComments, userName, targetPath)
            else:            
                potTarget = isTargetUser(user, targetIds)
                if(potTarget == classification.TARGET):
                    saveTextualData(usableComments, userName, targetPath)
                elif(potTarget == classification.NON_TARGET):
                    saveTextualData(usableComments, userName, nonTargetPath)
        return -1

def getTimeseriesUserData(user, targetIds, keywords, limit=1000, batchSize=400, fixedBatchSize=True, isTarget=True, existingUsers=[]):
    print('creating time series dataset')

    try:
        userName = user.name
    except AttributeError:
        return -1, -1, -1
    
    if(userExists(userName) and userName not in existingUsers):
        print('Working on user: ', userName)
        commentTimestamps = []
        commentsAdded = 0
        for comment in user.comments.new(limit=limit):
            if (commentsAdded >= batchSize):
                break
            try:
                if(hasattr(comment, 'created_utc')):
                    print('adding comment# ', str(len(commentTimestamps) + 1), 'for user: ', userName)
                    commentsAdded+= 1
                    commentTimestamps.append(comment.created_utc)
            except pe.TooManyRequests:
                print('too many requests, 5 minute timeout')
                time.sleep(300)
                continue
        if(len(commentTimestamps) >= batchSize or (not fixedBatchSize and len(commentTimestamps) >= 5)):
            if(isTarget):
                return commentTimestamps, True, userName
            else:            
                potTarget = isTargetUser(user, targetIds)
                if(potTarget == classification.TARGET):
                    return commentTimestamps, True, userName
                elif(potTarget == classification.NON_TARGET):
                    return commentTimestamps, False, userName
        else:
            print('user was deleted or already exists')
    return -1, -1, -1

def saveTextualData(dataArray, userName, path):
    filePath = os.path.join(path, userName + '.txt')
    seperator = '~\t~'
    commentString = seperator.join([entry for entry in dataArray])
    with open(filePath, 'w') as f:
        f.write(commentString)

def createTimeSeriesList(targetSubName, nonTargetSubNames, sizeGoal, comparisonSubNames, keywords, commentLimit, batchSize, fixedBatchSize, knownUsers=[]):
    targetIds = getTargetIds(comparisonSubNames)
    targetUsers = getSubUsers(targetSubName, 1000)
    timeseriesTuples = []
    nonTargetUsers2D = [getSubUsers(subName, round(1000/len(nonTargetSubNames))) for subName in nonTargetSubNames]
    nonTargetUsers = [item for sublist in nonTargetUsers2D for item in sublist]
    existingUsers = knownUsers

    for user in nonTargetUsers:
        if len(timeseriesTuples) < sizeGoal:
            timestamps, isTarget, userName = getTimeseriesUserData(user, targetIds, keywords, commentLimit, batchSize, fixedBatchSize, False, existingUsers)
            if (timestamps != -1 and userName != -1):
                existingUsers.append(userName)
                timeseriesTuples.append((timestamps, isTarget))
    while len(timeseriesTuples) < sizeGoal:
        nonTargetUsers = getSubUsers(targetSubName, 10)
        for user in nonTargetUsers:
            if (len(timeseriesTuples) >= sizeGoal):
                break
            timestamps, isTarget, userName = getTimeseriesUserData(user, targetIds, keywords, commentLimit, batchSize, fixedBatchSize, False, existingUsers)
            if (timestamps != -1 and userName != -1):
                existingUsers.append(userName)
                timeseriesTuples.append((timestamps, isTarget))
    for user in targetUsers:
        if len(timeseriesTuples) < sizeGoal * 2:
            timestamps, isTarget, userName = getTimeseriesUserData(user, targetIds, keywords, commentLimit, batchSize, fixedBatchSize, True, existingUsers)
            if (timestamps != -1 and userName != -1):
                existingUsers.append(userName)
                timeseriesTuples.append((timestamps, isTarget))
    while len(timeseriesTuples) < sizeGoal:
        nonTargetUsers = getSubUsers(targetSubName, 10)
        for user in nonTargetUsers:
            if (len(timeseriesTuples) >= sizeGoal * 2):
                break
            timestamps, isTarget, userName = getTimeseriesUserData(user, targetIds, keywords, commentLimit, batchSize, fixedBatchSize, True, existingUsers)
            if (timestamps != -1 and userName != -1):
                existingUsers.append(userName)
                timeseriesTuples.append((timestamps, isTarget))
    return timeseriesTuples, existingUsers


def fillDirectoryTextual(targetSubName, nonTargetSubNames, sizeGoal, path, testPath, comparisonSubNames, keywords, commentLimit, batchSize, fixedBatchSize):
    targetIds = getTargetIds(comparisonSubNames)
    targetUsers = getSubUsers(targetSubName, 1000)
    if (not os.path.exists(os.path.join(path))):
        os.mkdir(path)
    if (not os.path.exists(os.path.join(path, 'target'))):
        os.mkdir(os.path.join(path, 'target'))
    if (not os.path.exists(os.path.join(path, 'non_target'))):
        os.mkdir(os.path.join(path, 'non_target'))
    targetPath = os.path.join(path, 'target')
    nonTargetPath = os.path.join(path, 'non_target')
    nonTargetUsers2D = [getSubUsers(subName, round(1000/len(nonTargetSubNames))) for subName in nonTargetSubNames]
    nonTargetUsers = [item for sublist in nonTargetUsers2D for item in sublist]

    for user in nonTargetUsers:
        if (len(os.listdir(nonTargetPath)) >= sizeGoal):
            break
        getTextualUserData(user, path, testPath, targetIds, keywords, commentLimit, batchSize, fixedBatchSize, False)
    while(len(os.listdir(nonTargetPath)) < sizeGoal):
        nonTargetUsers = getSubUsers(targetSubName, 10)
        for user in nonTargetUsers:
            if (len(os.listdir(nonTargetPath)) >= sizeGoal):
                break
            getTextualUserData(user, path, testPath, targetIds, keywords, commentLimit, batchSize, fixedBatchSize, False)


    for user in targetUsers:
        if (len(os.listdir(targetPath)) >= sizeGoal):
            break
        getTextualUserData(user, path, testPath, targetIds, keywords, commentLimit, batchSize, fixedBatchSize, True)
    while(len(os.listdir(targetPath)) < sizeGoal):
        targetUsers = getSubUsers(targetSubName, 10)
        for user in targetUsers:
            if (len(os.listdir(targetPath)) >= sizeGoal):
                break
            getTextualUserData(user, path, testPath, targetIds, keywords, commentLimit, batchSize, fixedBatchSize, True)
