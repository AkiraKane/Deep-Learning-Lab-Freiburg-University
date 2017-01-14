import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import json
import random
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D
from keras.initializations import normal
from keras.optimizers import Adam,RMSprop
# custom modules
from utils import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable

disc_fact = 0.99
INITIAL_EPSILON = 0.9
FINAL_EPSILON = 0.1
opt = Options()
img_rows = img_cols= opt.cub_siz*opt.pob_siz
observe = 30000 # observe at least 30000 frames before training
explore = 10000 # annealing of the epsilon (random actions are more at first)
epsilon = INITIAL_EPSILON
test_steps = 3000 # test evey 3000 steps

# do some trajectories with only greedy actions and see the sum of the rewards is decreasing or not
def model():
    input_shape = (img_rows,img_cols,opt.hist_len)

    model = Sequential()
    model.add(Convolution2D(32,4,4,subsample=(2,2),activation='relu',
                            init= lambda shape, name: normal(shape, scale= 0.01, name= name),
                            border_mode='same',input_shape=input_shape))
    model.add(Convolution2D(64,2,2,subsample=(1,1),activation='relu',
                            init= lambda shape, name: normal(shape, scale= 0.01, name= name),
                            border_mode='same'))
    model.add(Convolution2D(64,2,2,subsample=(1,1),activation='relu',
                            init= lambda shape, name: normal(shape, scale= 0.01, name= name),
                            border_mode='same'))
    model.add(Flatten())
    model.add(Dense(256,activation='relu',init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Dense(opt.act_num,init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    model.compile(loss = 'mse',optimizer = Adam(lr=1e-6))
    return model


def append_to_hist(state, obs):
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# In contrast to your last exercise you DO NOT generate data before training
# instead the TransitionTable is build up while you are training to make sure
# that you get some data that corresponds roughly to the current policy
# of your agent
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization

sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
model = model()
# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None

# lets assume we will train for a total of 1 million steps
# this is just an example and you might want to change it
steps = 1 * 1000000
epi_step = 0
nepisodes = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)

# How to decreas the discount fator to make the algorithm converge ?

for step in range(steps):
    if (step < observe):
        print("Observing",step ,"from",observe,"steps")

    if state.terminal or epi_step >= opt.early_stop:
        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: here you would let your agent take its action
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # this just gets a random action
    epi_step+=1
    loss = 0

    ### TAKE AN ACTION ###
    if random.random() <= epsilon:  #random number between [0,1)
        print("Random")
        action = randrange(opt.act_num)
    else:
        print("Greedy")
        action = np.argmax(model.predict(np.transpose(state_with_history).reshape(1,img_rows,img_cols,opt.hist_len)))
    action_onehot = trans.one_hot_action(action)

    #annealing the epsilon
    if(epsilon>FINAL_EPSILON and step>observe):
        epsilon -= ((INITIAL_EPSILON-FINAL_EPSILON)/explore)

    #Take next step according to the action selected
    next_state = sim.step(action)
    # append state to history
    append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
    #add to the transition table
    trans.add(state_with_history.reshape(-1), action_onehot, next_state_with_history.reshape(-1), next_state.reward, next_state.terminal)

    # mark next state as current state
    state_with_history = np.copy(next_state_with_history)
    state = next_state
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # TODO: here you would train your agent
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #start training when at least a number of 'observe' frames are seen
    if(step >= observe):
        #Replay Experience
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
        #prediction for Q values from the CNN
        Q = model.predict_on_batch(state_batch.reshape(opt.minibatch_size,img_rows,img_cols,opt.hist_len))

        #prediction for Q values of next state from the CNN
        Qn =model.predict_on_batch(next_state_batch.reshape(opt.minibatch_size,img_rows,img_cols,opt.hist_len))

        #Double Q learning (Freazing the updates)
        # if((step % 1000) == 0):
        #     Qn =model.predict_on_batch(np.transpose(next_state_batch).reshape(opt.minibatch_size,img_rows,img_cols,opt.hist_len))

        #Target Q
        for i in range(0,opt.minibatch_size):
            Q[i,np.argmax(action_batch[i,:])] = ((1-terminal_batch[i,:]) * disc_fact* np.max(Qn[i,:])) + reward_batch[i,:]

        #reshape the input frames
        state_batch = state_batch.reshape(opt.minibatch_size,img_rows,img_cols,opt.hist_len)

        #Update the weights and calculate the loss accordingly
        loss = model.train_on_batch(state_batch,Q)

        # TODO every once in a while you should test your agent here so that you can track its performance
        if((step % test_steps) == 0):
            print("Testing")
            total_reward = 0
            for i in range(0,opt.eval_nepisodes):
                pass
            #     action = np.argmax(model.predict(np.transpose(state_with_history).reshape(1,img_rows,img_cols,opt.hist_len)))





        if opt.disp_on:
            if win_all is None:
                plt.subplot(121)
                win_all = plt.imshow(state.screen)
                plt.subplot(122)
                win_pob = plt.imshow(state.pob)
            else:
                win_all.set_data(state.screen)
                win_pob.set_data(state.pob)
            plt.pause(opt.disp_interval)
            plt.draw()
            print("Total Steps:",step)
            print('Episode %d, step %d, epsilon %.6f, action %d, reward %.5f' %
                  (nepisodes,epi_step,epsilon,action,state.reward))

    #There is a hyper parameter in the utilss calledd eval_nepisodes check it

    # 2. perform a final test of your model and save it
    # TODO
