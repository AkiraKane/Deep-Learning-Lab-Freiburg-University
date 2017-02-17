import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import json
import random
import argparse
import pickle
from timeit import default_timer as timer
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Flatten,Dropout
from keras.layers.convolutional import Convolution2D
from keras.initializations import normal
from keras.optimizers import Adam
from keras.regularizers import l2

# custom modules
from utils import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable
import logging


opt = Options()
img_rows = img_cols= opt.cub_siz*opt.pob_siz
test_steps = 200
steps = 2000000



#same model same hist len not same steps
#CNN model 
def model_create():
    input_shape = (img_rows,img_cols,opt.hist_len)

    model = Sequential()
    model.add(Convolution2D(128,8,8,subsample=(8,8),activation='relu',
                            init = lambda shape, name: normal(shape, scale= 0.01, name= name),
                            border_mode='same',input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128,activation='relu',init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Dense(opt.act_num,init=lambda shape, name: normal(shape, scale=0.01, name=name), activation = 'softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-5), sample_weight = None)
    return model  


def append_to_hist(state, obs):
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs
    
def play(args):
    # 0. initialization
    sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
    model = model_create()

    #continue training from a previous model
    ##model.load_weights(opt.weights_fil)
    
    # setup a transition table that is filled during training
    maxlen = opt.early_stop
    ##print('weights loaded to the model')
    trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len, maxlen)
    if args.mode == "train":
        print "Training mode"

        if opt.disp_on:
            win_all = None
            win_pob = None

        epi_step = 0
        nepisodes = 0

        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        state_with_history = np.zeros((opt.hist_len, opt.state_siz))
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
        loss = 0.
        reward_acc = 0.

        loss_list = []
        reward_acc_list = []
        epi_step_list = []
        reward_acc_new = []
        reward_acc_track = 0
        start = timer()
        #Training
        for step in range(steps):

            if state.terminal or epi_step >= opt.early_stop:
                state_batch, action_batch = trans.sample_minibatch(epi_step)
                state_batch = state_batch.reshape(epi_step,img_rows,img_cols,opt.hist_len)
                reward_sample_weight = np.zeros((epi_step,), dtype=np.float32) + reward_acc
                loss = model.train_on_batch(state_batch,action_batch, sample_weight = reward_sample_weight)
                print('Episode %d, step %d, total reward %.5f, loss %.8f' %
                                (nepisodes, epi_step, reward_acc, loss))

                # keep track of these values
                epi_step_list.append(epi_step)
                reward_acc_list.append(reward_acc)
##                loss_list.append(loss)
                
                
                epi_step = 0
                nepisodes += 1
                # reset the game
                state = sim.newGame(opt.tgt_y, opt.tgt_x)
                # and reset the history
                state_with_history[:] = 0
                append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
                next_state_with_history = np.copy(state_with_history)
                reward_acc = 0
                loss = 0
                trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len, maxlen)

            #Save the weights every now and then
            if(((step+1)%1000000) == 0):
                model.save_weights(opt.weights_fil, overwrite=True)
                print('Saved weights')
                with open(opt.network_fil, "w") as outfile:
                    json.dump(model.to_json(), outfile)
                

            epi_step+=1
            #sample an action from the policy network
            action = np.argmax(model.predict((state_with_history).reshape(1,img_rows,img_cols,opt.hist_len)))
            
            #one hot encoding
            action_onehot = trans.one_hot_action(action)

            #Take next step in the environment according to the action selected
            next_state = sim.step(action)
            
            # append state to history
            append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
            
            #add to the transition table
            trans.add(state_with_history.reshape(-1), action_onehot)
            # mark next state as current state
            state_with_history = np.copy(next_state_with_history)
            state = next_state
            reward_acc+= state.reward
            reward_acc_track += state.reward
            reward_acc_new.append(reward_acc_track)
            print "Total Steps:",step
            print('Episode %d, step %d, action %d, reward %.5f' %
                 (nepisodes,epi_step, action, state.reward))
            
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

        end = timer()
        sec = int(end-start)
        hours = int(sec/3600)
        rem = int(sec - (hours*3600))
        mins = rem/60
        rem = rem - (mins*60)
        secs = rem

        print 'Training time:',hours,':',mins,':',secs


        with open('episode_steps','wb') as f:
            pickle.dump(epi_step_list,f)
            print 'saved episode steps'
                    
        with open('accum_reward_episodes','wb') as f:
            pickle.dump(reward_acc_list,f)
            print 'saved accumulated reward for each episode'

##        with open('loss','wb') as f:
##            pickle.dump(loss_list,f)
##            print 'saved losses'

        with open('accum_reward_steps','wb') as f:
            pickle.dump(reward_acc_new,f)
            print 'saved accumulated reward for all steps'

                    
        #Save the weights
        model.save_weights(opt.weights_fil, overwrite=True)
        print('Saved weights')
        with open(opt.network_fil, "w") as outfile:
            json.dump(model.to_json(), outfile)


    ### run
    if args.mode == 'run':
        
        print "Running mode"
        model.load_weights(opt.weights_fil)
        print('weights loaded to the model')
        opt.disp_on = True
        win_all = None
        win_pob = None
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        state_with_history = np.zeros((opt.hist_len, opt.state_siz))
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
        epi_step = 0
        nepisodes = 0
        n_reached = 0.0
        reward_acc_test = 0
        reward_acc_list_test = []
        
        print('Test Phase')
        for test_step in range(test_steps):
            
            if state.terminal or epi_step > opt.early_stop:
                if(state.terminal):
                    print 'Episode:', (nepisodes +1),'agent reached'
                    n_reached+=1
                else:
                    print 'Episode:', (nepisodes +1),'agent failed'
                epi_step = 0
                nepisodes += 1
                # reset the game
                state = sim.newGame(opt.tgt_y, opt.tgt_x)
                # and reset the history
                state_with_history[:] = 0
                append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
                next_state_with_history = np.copy(state_with_history)
            
            epi_step+=1
            action = np.argmax(model.predict((state_with_history).reshape(1,img_rows,img_cols,opt.hist_len)))
            action_onehot = trans.one_hot_action(action)
            #Take next step according to the action selected
            next_state = sim.step(action)
            # append state to history
            append_to_hist(next_state_with_history, rgb2gray(next_state.pob).reshape(opt.state_siz))
        
            # mark next state as current state
            state_with_history = np.copy(next_state_with_history)
            state = next_state
            reward_acc_test+= state.reward
            
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
        print 'Agent reached the target', n_reached,'from',nepisodes,'episodes','(',(n_reached/nepisodes)*100,'%)'     



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode","-m",choices = {"train","run"}, default = "train", required = True)
    args = parser.parse_args()
    play(args)
