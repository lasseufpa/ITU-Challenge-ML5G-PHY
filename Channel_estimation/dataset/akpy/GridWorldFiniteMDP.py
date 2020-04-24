#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
'''
Implements the Grid World of Sutton & Barto's book
of Example 3.5: Gridworld, pag. 60
and
of Example 3.8: Solving the Gridworld, pag. 65
'''
from __future__ import print_function
import numpy as np
import itertools

from FiniteMDP import FiniteMDP


class GridWorldFiniteMDP(FiniteMDP):

    def postprocessing_MDP_step(self, history, printPostProcessingInfo):
        '''This method overrides its superclass equivalent and
        allows to post-process the results'''
        pass

    if False:
        def createActionsDataStructures(self):
            possibleActions = ['L', 'U', 'R', 'D']
            dictionaryGetIndex = dict()
            listGivenIndex = list()
            for uniqueIndex in range(len(possibleActions)):
                dictionaryGetIndex[possibleActions[uniqueIndex]] = uniqueIndex
                listGivenIndex.append(uniqueIndex)
            return dictionaryGetIndex, listGivenIndex

    def createStatesDataStructures(self,WORLD_SIZE):
        '''Defines the states. Overrides default method from superclass. WORLD_SIZE is the axis dimension, horizontal or vertical'''
        bufferStateList = list(itertools.product(np.arange(WORLD_SIZE), repeat=2))
        N = len(bufferStateList) #number of states
        stateListGivenIndex = list()
        stateDictionaryGetIndex = dict()
        uniqueIndex = 0
        # add states to both dictionary and its inverse mapping (a list)
        for i in range(N):
            stateListGivenIndex.append(bufferStateList[i])
            stateDictionaryGetIndex[bufferStateList[i]] = uniqueIndex
            uniqueIndex += 1
        if True:
            print('stateDictionaryGetIndex = ', stateDictionaryGetIndex)
            print('stateListGivenIndex = ', stateListGivenIndex)
        return stateDictionaryGetIndex, stateListGivenIndex

    def createFiniteMDP(self):
        '''Define the MDP process. Overrides default method from superclass.'''

        WORLD_SIZE = 5

        self.stateDictionaryGetIndex, self.stateListGivenIndex = self.createStatesDataStructures(WORLD_SIZE)

        A_POS = [0, 1]
        A_PRIME_POS = [4, 1]
        B_POS = [0, 3]
        B_PRIME_POS = [2, 3]
        self.discount = 0.9

        world = np.zeros((WORLD_SIZE, WORLD_SIZE))

        # left, up, right, down
        actions = ['L', 'U', 'R', 'D']
        self.actionDictionaryGetIndex, self.actionListGivenIndex = self.createActionsDataStructures(actions)

        S = WORLD_SIZE * WORLD_SIZE
        A = len(actions)

        # this is the original code from github
        actionProb = []
        for i in range(0, WORLD_SIZE):
            actionProb.append([])
            for j in range(0, WORLD_SIZE):
                actionProb[i].append(dict({'L': 0.25, 'U': 0.25, 'R': 0.25, 'D': 0.25}))
        # this is the original code from github
        nextState = []
        actionReward = []
        for i in range(0, WORLD_SIZE):
            nextState.append([])
            actionReward.append([])
            for j in range(0, WORLD_SIZE):
                next = dict()
                reward = dict()
                if i == 0:
                    next['U'] = [i, j]
                    reward['U'] = -1.0
                else:
                    next['U'] = [i - 1, j]
                    reward['U'] = 0.0

                if i == WORLD_SIZE - 1:
                    next['D'] = [i, j]
                    reward['D'] = -1.0
                else:
                    next['D'] = [i + 1, j]
                    reward['D'] = 0.0

                if j == 0:
                    next['L'] = [i, j]
                    reward['L'] = -1.0
                else:
                    next['L'] = [i, j - 1]
                    reward['L'] = 0.0

                if j == WORLD_SIZE - 1:
                    next['R'] = [i, j]
                    reward['R'] = -1.0
                else:
                    next['R'] = [i, j + 1]
                    reward['R'] = 0.0

                if [i, j] == A_POS:
                    next['L'] = next['R'] = next['D'] = next['U'] = A_PRIME_POS
                    reward['L'] = reward['R'] = reward['D'] = reward['U'] = 10.0

                if [i, j] == B_POS:
                    next['L'] = next['R'] = next['D'] = next['U'] = B_PRIME_POS
                    reward['L'] = reward['R'] = reward['D'] = reward['U'] = 5.0

                nextState[i].append(next)
                actionReward[i].append(reward)

        print('nextState = ', nextState)
        print('actionReward = ', actionReward)
        # now convert to our general and smarter :) format:
        self.nextStateProbability = np.zeros((S, A, S))
        self.rewardsTable = np.zeros((S, A, S))
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                nextsdic = nextState[i][j]  # this is a dictionary
                rdic = actionReward[i][j]  # another dictionary
                # get state index
                s = self.stateDictionaryGetIndex[(i, j)]
                for a in range(A):
                    (nexti, nextj) = nextsdic[actions[a]]
                    nexts = self.stateDictionaryGetIndex[(nexti, nextj)]
                    # After the agent chooses a state, the MDP “dynamics” is such that p(s’/s,a) is 1 to only one state and zero to the others
                    self.nextStateProbability[s, a, nexts] = 1
                    r = rdic[actions[a]]
                    self.rewardsTable[s, a, nexts] = r


if __name__ == '__main__':
    mdp = GridWorldFiniteMDP(discount=0.9)
    mdp.prettyPrint()

    #how to get only the p(s'/s) by marginalizing p(s'/a,s) (summing dimension i=1)
    onlyNextStateProbability = np.sum(mdp.nextStateProbability, 1)

    equiprobable_policy = mdp.getEquiprobableRandomPolicy()
    state_values, iteration = mdp.compute_state_values(equiprobable_policy, in_place=True)
    print('Equiprobable, iteration = ', iteration, ' state_values = ', np.round(state_values, 1))

    state_values, iteration = mdp.compute_optimal_state_values()
    print('Optimal, iteration = ', iteration, ' state_values = ', np.round(state_values, 1))

    action_values, iteration = mdp.compute_optimal_action_values()
    print('iteration = ', iteration, ' action_values = ', np.round(action_values, 1))

    #AK-TODO no need to be class method
    policy = mdp.convert_action_values_into_policy(action_values)
    mdp.prettyPrintPolicy(policy)

    mdp.run_MDP_for_given_policy(policy,maxNumIterations=100)

    mdp.execute_q_learning(maxNumIterations=100)