from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2,activity_l2
from keras.utils import np_utils
from keras.initializations import normal

import json
# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# NOTE:
# this script assumes you did generate your data with the get_data.py script
# you are of course allowed to change it and generate data here but if you
# want this to work out of the box first run get_data.py
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 0. initialization
opt = Options()
sim = Simulator(opt.map_ind, opt.cub_siz, opt.pob_siz, opt.act_num)
trans = TransitionTable(opt.state_siz, opt.act_num, opt.hist_len,
                             opt.minibatch_size, opt.valid_size,
                             opt.states_fil, opt.labels_fil)

# 1. train
######################################
# TODO implement your training here!
# you can get the full data from the transition table like this:
#
# # both train_data and valid_data contain tupes of images and labels
train_data = trans.get_train()
valid_data = trans.get_valid()

print "Train data shape:",train_data[0].shape
print "Train data labels shape:",train_data[1].shape
print "Valid data shape:",valid_data[0].shape
print "Valid data labels:",valid_data[1].shape

# 
# alternatively you can get one random mini batch line this
#
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
######################################

# 2. save your trained model

train_imgs = train_data[0]
train_labels = train_data[1]
val_imgs = valid_data[0]
val_labels = valid_data[1]

#define some parameters for the model
num_classes = 5
num_epochs = 10

batch_size = 32
num_filters = 32
pool_size = (2,2)
img_rows = img_cols= opt.cub_siz*opt.pob_siz


train_imgs = train_imgs.reshape(int(train_imgs.shape[0]),img_rows,img_cols,opt.hist_len)
val_imgs = val_imgs.reshape(int(val_imgs.shape[0]),img_rows,img_cols,opt.hist_len)
print("Train imgs shape:",train_imgs.shape)
print("Val imgs shape:",val_imgs.shape)
train_imgs = train_imgs.astype('float32')
val_imgs = val_imgs.astype('float32')

# i = 5004
# plt.imshow(train_imgs[i,:,:,0])
# plt.show()
# alternatively you can get one random mini batch line this
#
# for i in range(number_of_batches):
#     x, y = trans.sample_minibatch()
######################################

def model():
        input_shape = (img_rows,img_cols,opt.hist_len)
        model = Sequential()
        model.add(Convolution2D(32,3,3,subsample=(1,1),activation='relu',
                  init = lambda shape, name: normal(shape, scale= 0.01, name= name),
                  border_mode='same',input_shape=input_shape))

        model.add(Convolution2D(32,3,3,subsample=(1,1),activation='relu',
                  init = lambda shape, name: normal(shape, scale= 0.01, name= name),
                  border_mode='same',input_shape=input_shape))

        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.5))

        model.add(Convolution2D(64,3,3,subsample=(1,1),activation='relu',
                  init = lambda shape, name: normal(shape, scale= 0.01, name= name),
                  border_mode='same',input_shape=input_shape))

        model.add(Convolution2D(64,3,3,subsample=(1,1),activation='relu',
                  init = lambda shape, name: normal(shape, scale= 0.01, name= name),
                  border_mode='same',input_shape=input_shape))

        model.add(MaxPooling2D(pool_size = pool_size))
        model.add(Dropout(0.5))

        model.add(Convolution2D(128,3,3,subsample=(1,1),activation='relu',
                  init = lambda shape, name: normal(shape, scale= 0.01, name= name),
                  border_mode='same',input_shape=input_shape))

        model.add(Convolution2D(128,3,3,subsample=(1,1),activation='relu',
                  init = lambda shape, name: normal(shape, scale= 0.01, name= name),
                  border_mode='same',input_shape=input_shape))

        model.add(MaxPooling2D(pool_size = pool_size))
        #model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512,W_regularizer=l2(0.01),b_regularizer=l2(0.01),
                  init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        
        #model.add(Dropout(0.5))
        model.add(Dense(num_classes,W_regularizer=l2(0.01),b_regularizer=l2(0.01),
			      init=lambda shape, name: normal(shape, scale=0.01, name=name),activation='softmax'))
        
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

        return model

def main():
        agent = model()
        agent.fit(train_imgs,train_labels,batch_size=batch_size,nb_epoch=num_epochs,verbose=1,
                  validation_data=(val_imgs,val_labels))

        score = agent.evaluate(val_imgs,val_labels,verbose=0)
        print("Test Score:",score[0])
        print("Test accuracy:",score[1])

        # 2. save your trained model
        # serialize model to JSON
        agent.save_weights(opt.weights_fil, overwrite=True)
        print('Saved weights')
        with open(opt.network_fil, "w") as outfile:
                json.dump(agent.to_json(), outfile)

if __name__ == "__main__":
        main()

