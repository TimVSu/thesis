import os 
import datetime
import csv
import numpy as np

def createTimeSeriesDataset(userTuples, datasetPath, preprocessingFunctionName):
    if (os.path.exists(datasetPath)):
            os.remove(datasetPath)

    if preprocessingFunctionName == 'dayMinute':
        preprocessingFunction = findUsageDayMinutes
    elif preprocessingFunctionName == 'weekHour':
        preprocessingFunction = findUsageWeekHours
    elif preprocessingFunctionName == 'weekMinute':
        preprocessingFunction = findUsageWeekMinutes
    else:
        preprocessingFunction = findUsageDayHours
    targetList = []
    nonTargetList = []

    for timestamps, isTarget in userTuples:
        if isTarget:
            targetList.append([1.0] + zScoreNormalize(preprocessingFunction([float(timestamp) for timestamp in timestamps])))
        else:
            nonTargetList.append([0.0] + zScoreNormalize(preprocessingFunction([float(timestamp) for timestamp in timestamps])))

    with open(datasetPath, "w") as dataset:
        writer = csv.writer(dataset, delimiter='\t')
        writer.writerows(targetList + nonTargetList)


def findUsageDayHours(timeStampList):
    day = list(range(24))
    for point in timeStampList:
        hour = datetime.datetime.fromtimestamp(point).hour
        day[hour] = day[hour] + 1
    return day

def findUsageDayMinutes(timeStampList):
    day = list(range(1440))
    datetimes = [datetime.datetime.fromtimestamp(timestamp) for timestamp in timeStampList]
    minutesOfDay = [datetime.hour * 60 + datetime.minute for datetime in datetimes]
    return [sum([dayMinute == minuteOfDay for minuteOfDay in minutesOfDay]) for dayMinute in day]

def findUsageWeekHours(timeStampList):
    day = list(range(168))
    datetimes = [datetime.datetime.fromtimestamp(timestamp) for timestamp in timeStampList]
    minutesOfDay = [datetime.weekday() * 24 + datetime.hour for datetime in datetimes]
    return [sum([dayMinute == minuteOfDay for minuteOfDay in minutesOfDay]) for dayMinute in day]

def findUsageWeekMinutes(timeStampList):
    day = list(range(1440))
    datetimes = [datetime.datetime.fromtimestamp(timestamp) for timestamp in timeStampList]
    minutesOfDay = [datetime.weekday() * 24 + datetime.hour * 60 + datetime.minute for datetime in datetimes]
    return [sum([dayMinute == minuteOfDay for minuteOfDay in minutesOfDay]) for dayMinute in day]

def zScoreNormalize(data):
    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        raise ValueError("Standard deviation of data is zero")

    normalized_data = [(x - mean) / std for x in data]

    return normalized_data