import csv  # imports csv
import pandas as pd  # imports panda but changes name to pd
import tensorflow as tf
import numpy as np
import math
# Profile Columns'age', 'body_type', 'diet', 'drinks', 'drugs', 'education', 'essay0',
# 'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7',
# 'essay8', 'essay9', 'ethnicity', 'height', 'income', 'job',
# 'last_online', 'location', 'offspring', 'orientation',  'pets',
# 'religion', 'sex', 'sign', 'smokes', 'speaks', 'status'
# print(profiles['age'])
class Person: # This is the person class
    def __init__(self,age,sex,height,income): # this is alot of data to parse
        self.age = age  # starting with these basic ones
        self.sex = sex
        self.height = height
        self.income = income
       # words = pd.read_table(glove, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE) # Glove
    def vec(w):
        return words.loc[w].as_matrix()
class Population: #this is made up of a population of people to match
    def __init__(self,csvfile):
        self.people = addPerson(self,csvfile) #this is an array of persons
    def addPerson(self,csvfile):
        'This method adds people to the population by taking in a csvfile ofof data'
        self.people.insert(Person)
    def remPerson(self,Person):
        self.people.remove(Person)
class NeuralNetwork:
    def __init__(self,inputNodes,outputNodes, middleLayer):
        self.inputNodes = inputNodes
        self.outputNodes = outputNodes
        self.middleLayer = middleLayer
    def sigmoid(self,x):
        return 1/(1+math.exp(-x))
class UserInterface:
    def __init__(self):
        self.startWelcome()
        self.loadProfiles()
        self.rateProfiles()
    def startWelcome():
        print("Welcome to Matching Algorithm for Dating")
        print("Soon you will see a couple of profiles and you will judge them")
        print("Using a scale from 1-10")
        print("Our Algorithm will do some work then give you some more ")
        print("Eventually you will see the options becoming more attuned for your taste")
    def loadProfiles():
        with open('profiles.csv', 'rt') as csvfile:  # opens the file of profile info
            profilesreader = csv.reader(csvfile, delimiter=',' , quotechar='|')
            profiles = pd.read_csv(csvfile)
    def rateProfiles(self,Population):
        stateSpace = []
        for i in range(10):
            stateSpace.append(random.choice(Population))
        for person in stateSpace:
            print("Please enter a number between 1 and 10")
            r,S,a = input("This profile has"+"Age:"+person.Age+"sex:"+person.sex+"h"+person.height),person,action,


# Algorithm
# 1.Randomly Select People while error is > .2
# 2.Convert Person into a vector of attributes using glove
# 3.Send Vector through a neural network
# 4.User gives feedback
