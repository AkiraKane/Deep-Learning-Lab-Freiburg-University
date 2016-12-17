from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2,activity_l2
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json

# custom modules
from utils     import Options
from simulator import Simulator
from transitionTable import TransitionTable
from matplotlib import pyplot as plt
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
print("Train data shape:",train_data[0].shape)
print("Train data labels shape:",train_data[1].shape)
print("Valid data shape:",valid_data[0].shape)
print("Valid data labels:",valid_data[1].shape)

train_imgs = train_data[0]
train_labels = train_data[1]
val_imgs = valid_data[0]
val_labels = valid_data[1]

#define some parameters for the model
num_classes = 5
num_epochs = 10
kernel_size = (3,3)
batch_size = 128
num_filters = 32
pool_size = (2,2)


train_imgs = train_imgs.reshape(int(train_imgs.shape[0]),50,50,1)
val_imgs = val_imgs.reshape(int(val_imgs.shape[0]),50,50,1)
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

input_shape = (50,50,1)
network = Sequential()
network.add(Convolution2D(num_filters,kernel_size[0],kernel_size[1],activation='relu',
                        border_mode='same',input_shape = input_shape))

network.add(Convolution2D(num_filters,kernel_size[0],kernel_size[1],activation='relu',
                        border_mode='same'))
network.add(MaxPooling2D(pool_size=pool_size))
network.add(Dropout(0.25))

network.add(Convolution2D(num_filters*2,kernel_size[0],kernel_size[1],activation='relu',
                        border_mode='same'))
network.add(Convolution2D(num_filters*2,kernel_size[0],kernel_size[1],activation='relu',
                        border_mode='same'))
network.add(MaxPooling2D(pool_size= pool_size))
network.add(Dropout(0.25))

network.add(Convolution2D(num_filters*3,kernel_size[0],kernel_size[1],activation='relu',
                        border_mode='same'))
network.add(Convolution2D(num_filters*3,kernel_size[0],kernel_size[1],activation='relu',
                        border_mode='same'))
network.add(MaxPooling2D(pool_size = pool_size))
network.add(Dropout(0.25))

network.add(Flatten())
network.add(Dense(256,W_regularizer=l2(0.01),b_regularizer=l2(0.01)))
network.add(Dropout(0.5))

network.add(Dense(num_classes,W_regularizer=l2(0.01),b_regularizer=l2(0.01),activation='softmax'))

network.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])

network.fit(train_imgs,train_labels,batch_size=batch_size,nb_epoch=num_epochs,verbose=1,
          validation_data=(val_imgs,val_labels))

score = network.evaluate(val_imgs,val_labels,verbose=0)
print("Test Score:",score[0])
print("Test accuracy:",score[1])


# 2. save your trained model
# serialize model to JSON
network_json =  network.to_json()
with open(opt.network_fil,"w") as json_file:
    json_file.write(network_json)

#serialize the weights to HDF5
network.save_weights(opt.weights_fil)
print("Saved model to disk")