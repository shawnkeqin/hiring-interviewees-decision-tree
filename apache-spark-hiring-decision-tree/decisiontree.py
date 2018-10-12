from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array


conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf = conf)

def binary(decision):
    if (decision == 'Y'):
        return 1
    else:
        return 0

def EducationLevel(degree):
    if (degree == 'BEng'):
        return 1
    elif (degree =='MSc'):
        return 2
    elif (degree == 'PhD'):
        return 3
    else:
        return 0


def createLabeledPoints(fields):
    yearsOfExperience = int(fields[0])
    employed = binary(fields[1])
    pastEmployers = int(fields[2])
    educationLevel = EducationLevel(fields[3])
    schoolRanking = binary(fields[4])
    internshipExperience = binary(fields[5])
    hiredStatus = binary(fields[6])

    return LabeledPoint(hiredStatus, array([yearsOfExperience, employed,
        pastEmployers, educationLevel, schoolRanking, internshipExperience]))


rawData = sc.textFile("C:/Users/User/Contacts/Desktop/apache-spark-hiring-decision-tree/PastHires.csv")
header = rawData.first()
rawData = rawData.filter(lambda x:x != header)


csvData = rawData.map(lambda x: x.split(","))

trainingData = csvData.map(createLabeledPoints)


testCandidates = [ array([6, 1, 2, 1, 0, 0])]
testData = sc.parallelize(testCandidates)


model = DecisionTree.trainClassifier(trainingData, numClasses=2,
                                     categoricalFeaturesInfo={1:2, 3:4, 4:2, 5:2},
                                     impurity='gini', maxDepth=5, maxBins=32)


predictions = model.predict(testData)
print('Do We Hire Him/Her? :')
results = predictions.collect()
for result in results:
    print(result)


print('Decision tree model:')
print(model.toDebugString())
