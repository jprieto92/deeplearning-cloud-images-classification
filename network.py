#################################
### JAVIER PRIETO - 100307011 ###
#################################

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from sklearn.utils import shuffle
import numpy as np
import pickle

fil = open('dataset.p', 'r')
dataset = pickle.load(fil)
fil.close()

#############################
######### VARIABLES #########
#############################

# There are all 717 numpy arrays

dates = dataset['dates']                # dates when the pictures were taken
cloudtypes = dataset['cloudtypes']      # cloudtypes (labels)
images = dataset['images']              # images to use (256 x 256 x 3)
N = dataset['N']                        # number of samples (717)

#############################
######## HANDLE DATA ########
#############################

X = [] # images
Y = [] # labels

for i in range(N):
    image = images[i]
    label = cloudtypes[i]

    # combine cirros and ciroostratos into class 1

    if 'cirros' == label or 'cirrostratos' == label:
        X.append(image)
        Y.append([1,0])

    # discard multinube
    elif 'multinube' == label:
        pass

    # else -> class 2
    else:
        X.append(image)
        Y.append([0,1])

N = len(X)
X, Y = shuffle(X, Y) # shuffle

# convert into numpy arrays

X = np.array(X)
Y = np.array(Y)

#############################
######### DATASETS ##########
#############################

X_train = X[:int(N*0.7)]           # X training images (70%)
Y_train = Y[:int(N*0.7)]           # Y training labels (30%)

X_test  = X[int(N*0.7):]           # X test labels (30%)
Y_test  = Y[int(N*0.7):]           # Y test labels (70%)

#############################
########### MODEL ###########
#############################

input_shape = (256, 256, 3)
epochs = 10
batch_size = 10
num_classes = len(Y[0])

# Create model

model = Sequential()

# Convolutional layers

model.add(Conv2D(64, kernel_size=(11,11), strides=4, padding='same', input_shape=input_shape, data_format='channels_last', activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4), strides=4, data_format='channels_last'))
model.add(BatchNormalization(axis=1))

model.add(Conv2D(32, kernel_size=(7,7), strides=2, padding='same', data_format='channels_last', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=1, data_format='channels_last'))
model.add(BatchNormalization(axis=1))

model.add(Conv2D(32, kernel_size=(7,7), strides=2, padding='same', data_format='channels_last', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=1, data_format='channels_last'))
model.add(BatchNormalization(axis=1))

model.add(Conv2D(16, kernel_size=(5,5), strides=2, padding='same', data_format='channels_last', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=1, data_format='channels_last'))
model.add(BatchNormalization(axis=1))

# Flat feature map for the dense layers

model.add(Flatten())

# Dense layers

model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.4))

# Output layers

model.add(Dense(num_classes, activation='softmax'))

# Print the summary of the model

model.summary()

# Classic Stochastic Gradient Descent (0.5 learning rate, 5*10-7 decay, 0 momentum)

opt = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.00005, nesterov=False)

# Compile the model

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Training

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, shuffle=True)

# Test

scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
