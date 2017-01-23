import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import json
import random
from timeit import default_timer as timer
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Flatten,Dropout
from keras.layers.convolutional import Convolution2D
from keras.initializations import normal
from keras.optimizers import Adam,RMSprop
# custom modules
from utils import Options, rgb2gray
from simulator import Simulator
from transitionTable import TransitionTable

disc_fact = 0.99
INITIAL_EPSILON = 0.7
FINAL_EPSILON = 0.2
opt = Options()
img_rows = img_cols= opt.cub_siz*opt.pob_siz
observe = 50000 # observe at least 50000 frames before training
explore = 100000 # annealing of the epsilon
epsilon = INITIAL_EPSILON
test_steps = 2000
steps = 600000

#53
#CNN model 
def model():
    input_shape = (img_rows,img_cols,opt.hist_len)

    model = Sequential()
    model.add(Convolution2D(32,8,8,subsample=(4,4),activation='relu',
                            init= lambda shape, name: normal(shape, scale= 0.01, name= name),
                            border_mode='same',input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(32,activation='relu',init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Dense(opt.act_num,init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    model.compile(loss = 'mse',optimizer = Adam(lr=1e-6))
    return model  

def target_model():
    input_shape = (img_rows,img_cols,opt.hist_len)

    model = Sequential()
    model.add(Convolution2D(32,8,8,subsample=(4,4),activation='relu',
                            init= lambda shape, name: normal(shape, scale= 0.01, name= name),
                            border_mode='same',input_shape=input_shape))

    model.add(Flatten())
    model.add(Dense(32,activation='relu',init=lambda shape, name: normal(shape, scale=0.01, name=name)))
    model.add(Dense(opt.act_num,init=lambda shape, name: normal(shape, scale=0.01, name=name)))

    model.compile(loss = 'mse',optimizer = Adam(lr=1e-6))
    return model  


def append_to_hist(state, obs):
    for i in range(state.shape[0]-1):
        state[i, :] = state[i+1, :]
    state[-1, :] = obs
    

# 0. initialization
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
model = model()
target_model = target_model()


#continue training from a previous model
#model.load_weights(opt.weights_fil)
#print('weights loaded to the model')

# setup a large transitiontable that is filled during training
maxlen = 100000
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                        opt.minibatch_size, maxlen)

if opt.disp_on:
    win_all = None
    win_pob = None

epi_step = 0
nepisodes = 0

state = sim.newGame(opt.tgt_y, opt.tgt_x)
state_with_history = np.zeros((opt.hist_len, opt.state_siz))
append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
next_state_with_history = np.copy(state_with_history)
loss = 0
reward_acc_train = 0
reward_acc_list_train = []



start = timer()
#Training
for step in range(steps):
    if (step < observe):
        print("Observing",step ,"from",observe,"steps")

    if state.terminal or epi_step > opt.early_stop:
        epi_step = 0
        nepisodes += 1
        # reset the game
        state = sim.newGame(opt.tgt_y, opt.tgt_x)
        # and reset the history
        state_with_history[:] = 0
        append_to_hist(state_with_history, rgb2gray(state.pob).reshape(opt.state_siz))
        next_state_with_history = np.copy(state_with_history)
        
    epi_step+=1

    ### TAKE AN ACTION ###
    if random.random() <= epsilon or step<observe:
        #Random
        action = randrange(opt.act_num)
    else:
        #Greedy
        action = np.argmax(model.predict((state_with_history).reshape(1,img_rows,img_cols,opt.hist_len)))
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
    reward_acc_train += state.reward
    reward_acc_list_train.append(reward_acc_train)

    #start training when at least a number of 'observe' frames are seen
    if((step > observe)): #and (step%train_steps == 0)):
        #Replay Experience
        state_batch, action_batch, next_state_batch, reward_batch, terminal_batch = trans.sample_minibatch()
        #reshape the input frames
        state_batch = state_batch.reshape(opt.minibatch_size,img_rows,img_cols,opt.hist_len)
        #reshape the next input frames
        next_state_batch = next_state_batch.reshape(opt.minibatch_size,img_rows,img_cols,opt.hist_len)
        
        #prediction for Q values w.r.t the model
        Q = model.predict_on_batch(state_batch) 
        
        #prediction for Q values of next state w.r.t. the model target_model (OLD FIXED WEIGHTS)
        Qn = target_model.predict_on_batch(next_state_batch) 

        #Switch target
        #Copy the weights of the new model to the target model every 10000 steps
        if (step % 10000) == 0:
            print('Copying weights to the old model')
            target_model.set_weights(model.get_weights())
        
        #compute Q-learning targets w.r.t. old FIXED weights
        for i in range(0,opt.minibatch_size):
            Q[i,np.argmax(action_batch[i])] = ((1-terminal_batch[i]) * disc_fact* np.max(Qn[i])) + reward_batch[i]
        
        #Update the weights
        loss = model.train_on_batch(state_batch,Q)
        

    print "Total Steps:",step
    print('Episode %d, step %d, epsilon %.6f, action %d, reward %.5f, loss %.5f' %
            	(nepisodes,epi_step,epsilon,action,state.reward,loss))
        

    
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

#Save the weights
model.save_weights(opt.weights_fil, overwrite=True)
print('Saved weights')
with open(opt.network_fil, "w") as outfile:
    json.dump(model.to_json(), outfile)


#Testing
opt.disp_on = True
win_all = None
win_pob = None
action = np.argmax(model.predict((state_with_history).reshape(1,img_rows,img_cols,opt.hist_len)))
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
    reward_acc_list_test.append(reward_acc_test)
    
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




plt.figure(2)
plt.subplot(2,1,1)
plt.plot(reward_acc_list_train)
plt.title('Training steps')
plt.ylabel('Accumulated Reward')
plt.xlabel('Number of steps')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(reward_acc_list_test)
plt.title('Testing steps')
plt.ylabel('Accumulated Reward')
plt.xlabel('Number of steps')
plt.grid(True)

plt.show()



