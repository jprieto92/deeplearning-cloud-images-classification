'''
######################################
#  JAVIER PRIETO CEPEDA - 100307011  #
######################################
'''


from sklearn.cross_validation import StratifiedKFold # for cross-validation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from sklearn.utils import shuffle
import numpy as np
import pickle

'''
######################################
# Load the data and generate X and Y #
######################################
'''
def load_data():

    with open('dataset.p', 'r') as fil:     # open serialized file
        dataset = pickle.load(fil)          # load pickled dict

    N = dataset['N']                        # number of samples (717)  
    del dataset['N']                        # delete N from the dict

    dataset = zip(*dataset.values())        # dict -> transposed list
    dataset.sort(key=lambda x: x[2])        # sort by date

    X = [] # images
    Y = [] # labels

    for i in dataset:
        label = i[0] # label of sample i
        image = i[1] # pixels of sample i

        # combine cirros and ciroostratos into class 1

        if 'cirros' == label or 'cirrostratos' == label:
            X.append(image)
            Y.append(1)

        # discard multinube
        elif 'multinube' == label:
            pass

        # else -> class 2
        else:
            X.append(image)
            Y.append(0)

    N = len(X)           # number of samples after generating the datasets
    X, Y = shuffle(X, Y) # shuffle

    # convert into numpy arrays

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

'''
######################################
########## Create the model ##########
######################################
'''
def create_model(input_shape, num_classes):

    # Initialize the model

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

    #model.add(Dropout(0.4))

    # Output layers

    model.add(Dense(num_classes, activation='sigmoid'))

    # Print the summary of the model

    #model.summary()

    # Classic Stochastic Gradient Descent (0.5 learning rate, 5*10-7 decay, 0 momentum)

    opt = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.00005, nesterov=False)

    # Compile the model

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

'''
######################################
########### Train and test ###########
######################################
'''
def train_and_test_model(model, X_train, Y_train, X_test, Y_test):

    epochs = 10
    batch_size = 10
 
    # Training

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=batch_size, shuffle=True)

    # Test

    scores = model.evaluate(X_test, Y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    return scores

'''
######################################
################ Main ################
######################################
'''
if __name__ == "__main__":

    input_shape = (256, 256, 3)				     # Input shape
    n_folds = 5                                              # number of folds
    X, Y = load_data()                                       # load X and Y
    num_classes = 1                                          # Number of classes (outputs)
    skf = StratifiedKFold(Y, n_folds=n_folds, shuffle=False) # Generate train/test indices to split data in train test sets

    cvscores = []

    for i, (train, test) in enumerate(skf):
        print "Running Fold", i+1, "/", n_folds
        model = None                                                               # Clearing the CNN
        model = create_model(input_shape, num_classes)                             # Create the model
        scores = train_and_test_model(model, X[train], Y[train], X[test], Y[test]) # Train and test
        cvscores.append(scores[1] * 100)

    print("\n\n###################\n\nFinal results:\n")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
