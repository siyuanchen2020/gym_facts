import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy
import os
from random import randint
from stable_baselines3.common.env_checker import check_env



class BasicEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(81000)
        self.observation_space = spaces.Discrete(2)

    def step(self, action):
        # if we took an action, we were in state 1
        state = 0

        '''
        for a in range(1,4):
            for b in range(1,4):
                for c in range(1,4):
                    for d in range(1,4):
                        for e in range(1,11):
                            for f in range(1,11):
                                for g in range(1,11):
                                    input_var = numpy.array([a,b,c,d,e,f,g])
                                    #list_var.append(input_var)
                                    numpy.savetxt('input.txt', input_var)

                                    os.system("xsim-runner.exe --model LawMcComasMOPs.xml --input input.txt --output_txt output_Law.txt")

                                    with open('output_Law.txt') as my_file:
                                        # Throughput, Work-In-Process, Parts-Produced, and Lead-Time
                                        output_array = my_file.readlines()
                                    # get the throughput
                                    throughput = float(output_array[0])
                                    reward = (200 * throughput * 720) - 25000 * (input_var[0] + 1 + 1 + 1) - 1000 * (
                                                1 + 1 + 1)
        '''


        a = randint(1,3)
        b = randint(1,3)
        c = randint(1,3)
        d = randint(1,3)
        e = randint(1,10)
        f = randint(1,10)
        g = randint(1,10)

        input_var = numpy.array([a,b,c,d,e,f,g])
        numpy.savetxt('input.txt', input_var)

        os.system(
            "xsim-runner.exe --model LawMcComasMOPs.xml --input input.txt --output_txt output_Law.txt")

        with open('output_Law.txt') as my_file:
            # Throughput, Work-In-Process, Parts-Produced, and Lead-Time
            output_array = my_file.readlines()
        # get the throughput
        parts_produced = float(output_array[2])
        reward = (200 * parts_produced) - 25000 * (input_var[0] + input_var[1] + input_var[2] + input_var[3]) - 1000 * (input_var[4] + input_var[5] + input_var[6])
        
        print(input_var)
        #standardize the reward
        reward = (reward - 300000)/300000


        state = 1
        # regardless of the action, game is done after a single step
        done = True

        info = {}

        return state, reward, done, info

    def reset(self):
        state = 0
        return state

    def render(self, mode='human'):
        pass

    def close(self):
        pass


#check_env(BasicEnv,warn= True, skip_render_check=True)