# This is forked from Rodney Thomas's notebook [] and edited by Abdelrahman Matar to test Xception model from Keras pretrained on Imagenet.
from __future__ import division

import six
import numpy as np
import pandas as pd
import cv2
import glob
import random

np.random.seed(2016)
random.seed(2016)

import keras
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger
from keras.callbacks import Callback
from keras import backend as K
from keras.applications.xception import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint


# ## Removes autoscroll throughout process
get_ipython().run_cell_magic(u'javascript', u'', u'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# ## Global Declarations

conf = dict()

# How many patients will be in train and validation set during training. Range: (0; 1)
conf['train_valid_fraction'] = 0.75

# Batch size for CNN [Depends on GPU and memory available]
conf['batch_size'] = 1

# Number of epochs for CNN training
#conf['nb_epoch'] = 200
conf['nb_epoch'] = 5

# Early stopping. Stop training after epochs without improving on validation
conf['patience'] = 3

# Shape of image for CNN (Larger the better, but you need to increase CNN as well)
#conf['image_shape'] = (4160,4128)
#conf['image_shape'] = (2080,2064)
#conf['image_shape'] = (1024,1024)
conf['image_shape'] = (299,299) #to fit Xception

# ## Batch Generator for model fit_generator
def batch_generator_train(files, batch_size):
    number_of_batches = np.ceil(len(files)/batch_size)
    counter = 0
    random.shuffle(files)
    while True:
        batch_files = files[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []
        for f in batch_files:
            image = cv2.imread(f)
            image = cv2.resize(image, conf['image_shape'])

            cancer_type = f[20:21] # relies on path lengths that is hard coded below
            if cancer_type == '1':
                mask = [1, 0, 0]
            elif cancer_type == '2':
                mask = [0, 1, 0]
            else:
                mask = [0, 0, 1]

            image_list.append(image)
            mask_list.append(mask)
        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)

        yield image_list, mask_list

        if counter == number_of_batches:
            random.shuffle(files)
            counter = 0


# ## Hardcoded paths to training files. Note that "additional" has been renamed to "add01" since the path lengths must be the same for substring extraction.
# file paths to training and additional samples
filepaths = []
base_file_path = '/home/cse/abdelrahmanML/project/cervix/'
filepaths.append(base_file_path + 'train/Type_1/')
filepaths.append(base_file_path + 'train/Type_2/')
filepaths.append(base_file_path + 'train/Type_3/')


# ## Get a list of all training files
allFiles = []

for i, filepath in enumerate(filepaths):
	print "filepath= ", filepath
	files = glob.glob(filepath + '*.jpg')
	print ("len(files)= ", len(files))
	allFiles = allFiles + files

print('Loaded Files: {}'.format(len(allFiles)))
# ## Split data into training and validation sets
split_point = int(round(conf['train_valid_fraction']*len(allFiles)))

random.shuffle(allFiles)

train_list = allFiles[:split_point]
valid_list = allFiles[split_point:]
print('Train patients: {}'.format(len(train_list)))
print('Valid patients: {}'.format(len(valid_list)))


# ## Testing model generator with Xception Network
print('Create and compile model...')

nb_classes = 3
img_rows, img_cols = conf['image_shape'][1], conf['image_shape'][0]
img_channels = 3

base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet', pooling = 'avg')
x = base_model.output
predictions = Dense(3, activation='softmax')(x)
model = Model(input=base_model.input, output=predictions)

#model.summary()

for layer in model.layers:
   layer.trainable = True

#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='hinge',optimizer='adadelta',metrics=['accuracy'])
SGD = SGD(lr=1e-4, momentum=0.8, nesterov=True)
model.compile(optimizer=SGD, loss='categorical_crossentropy', metrics=['accuracy'])

class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))

myCallbacks = [
    EarlyStopping(monitor='val_loss', patience=conf['patience'], verbose=0),
    ModelCheckpoint('./weights/w1.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=0, mode='auto', period= 1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=0, min_lr=1e-6, verbose=1),
    SGDLearningRateTracker(),
]

print('Fit model...')

fit = model.fit_generator(generator=batch_generator_train(train_list, conf['batch_size']),
			steps_per_epoch=len(train_list)//conf['batch_size'],
			nb_epoch=conf['nb_epoch'],
			#samples_per_epoch=len(train_list),
			#samples_per_epoch=len(train_list)//conf['batch_size'],
			validation_data=batch_generator_train(valid_list, conf['batch_size']),
			validation_steps=len(valid_list), #//conf['batch_size'],
			verbose=1,
			callbacks=myCallbacks)

# ## Create submission files with prediction for submission
#from keras.models import load_model
#model = load_model('cervical_best.hdf5')

sample_subm = pd.read_csv(base_file_path + "sample_submission.csv")
ids = sample_subm['image_name'].values

for id in ids:
    print('Predict for image {}'.format(id))
    files = glob.glob(base_file_path + "test/" + id)
    image_list = []
    for f in files:
        image = cv2.imread(f)
        image = cv2.resize(image, conf['image_shape'])
        image_list.append(image)
        
    image_list = np.array(image_list)

    predictions = model.predict(image_list, verbose=1, batch_size=1)

    sample_subm.loc[sample_subm['image_name'] == id, 'Type_1'] = predictions[0,0]
    sample_subm.loc[sample_subm['image_name'] == id, 'Type_2'] = predictions[0,1]
    sample_subm.loc[sample_subm['image_name'] == id, 'Type_3'] = predictions[0,2]
    
sample_subm.to_csv("subm.csv", index=False)

