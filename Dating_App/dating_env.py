from gym import logger
import numpy as np
import csv
import pandas as pd
import gym
import random
from gym import error, spaces
from gym.utils import closer

from main import Person

class DatingEnv():
    """ This will be the evniroment that the agent will interact with
         it is based off of environments from the OpenAI gym library"""
    metadata = {'render.modes' : ['human']}
    def __init__(self):
        high = np.array([np.inf]*8)
        self.observation_space = spaces.Box(-high,high)
        self.action_space = spaces.Box(low=0, high=28, shape=(179838,))
        self.Population = []
        self.measurePop = []
        self.loadProfiles()
    def reset(self):
         print("Welcome to Matching Algorithm for Dating")
         print("Soon you will see a couple of profiles and you will judge them")
         print("Using a scale from 1-10")
         print("Our Algorithm will do some work then give you some more ")
         print("Eventually you will see the options becoming more attuned for your taste")
         return self.__getObservation()
    def step(self, action):
        self.__performAction(action)
        return self.__getObservation(), self.__getReward(), self.__getStatus,{ }
    def __performAction(self, input_actions):
        stateSpace = []
        print(input_actions)
        x = np.mean(input_actions)
        number = x.astype(int)
        for i in range(number):
            stateSpace.append(random.choice(self.Population))
        self.measurePop = stateSpace
    def __getObservation(self):
        """Returns the population that the agent has to choose from"""
        darray = []

        for person in self.Population:
            x = np.array([person.age,person.income,person.height])
            darray.append(x)


        farray = np.array(darray)
        return farray
    def __getReward(self):
        """Returns the users score on the profile it was just shown"""
        print("Please enter a score from 1-10 for each Profile")
        reward = 0
        for person in self.measurePop:
            reward += int(input("Age: "+str(person.age)+" Sex "+ person.sex+" Height:"+str(person.height)+ " Income:"+str(person.income)))
        return reward
    def __getStatus(self):
        """ If the user wants to keep rating return True else false"""
        return True
    def loadProfiles(self):
        with open('profiles.csv', 'rt') as csvfile:  # opens the file of profile info
            profilesreader = csv.reader(csvfile, delimiter=',' , quotechar='|')
            profiles = pd.read_csv(csvfile)
            for index, row in profiles.iterrows():
                person=Person(row['age'],row['sex'],row['height'],row['income'])
                self.Population.append(person)
