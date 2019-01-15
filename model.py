import csv
import numpy as np
from scipy import ndimage

# reading the lines in the log csv file for later identification of filename
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
# arrays for images for X_train and steering angle measurements for y_train
images = []
angles = []

for line in lines:
    name = '../data/IMG/'+line[0].split('/')[-1]
    center_image = ndimage.imread(name)
    # augmenting the data by flipping the image
    center_image_flipped = np.fliplr(center_image) 
    center_angle = float(line[3])
    # augmenting the data by "flipping" the steering angle by multiplying with -1.0
    center_angle_flipped = -center_angle
    images.append(center_image)
    images.append(center_image_flipped)
    angles.append(center_angle)
    angles.append(center_angle_flipped)

X_train = np.array(images)
y_train = np.array(angles)

# keras imports
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
#from keras.layers.pooling import MaxPooling2D
#from keras.utils import plot_model

# using the NVIDIA model as network to train
model = Sequential()
# normalizing and centering the data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
# cropping the images from the top part where the sky and trees are and the bottom part with the car hood
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
#model.add(Dropout(0.2))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
#model.add(Dropout(0.2))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
#model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#plot_model(model, to_file='model.png')

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2)

model.save('model.h5')

# alternative code for using fit_generator, but this ran much slower and wasn't needed as I didn't run into memory problems

#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle
#train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#def generator(samples, batch_size=128):
#    num_samples = len(samples)
#    while 1: # Loop forever so the generator never terminates
#        shuffle(samples)
#        for offset in range(0, num_samples, batch_size):
#            batch_samples = samples[offset:offset+batch_size]
#            images = []
#            angles = []
#            for batch_sample in batch_samples:
#                name = '../data/IMG/'+batch_sample[0].split('/')[-1]
#                center_image = ndimage.imread(name)
#                center_image_flipped = np.fliplr(center_image)
#                center_angle = float(batch_sample[3])
#                center_angle_flipped = -center_angle
#                images.append(center_image)
#                images.append(center_image_flipped)
#                angles.append(center_angle)
#                angles.append(center_angle_flipped)

#            X_train = np.array(images)
#            y_train = np.array(angles)
#            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
#train_generator = generator(train_samples, batch_size=128)
#validation_generator = generator(validation_samples, batch_size=128)

#model.fit_generator(train_generator, steps_per_epoch=len(train_samples), \
#                    validation_data=validation_generator, validation_steps=len(validation_samples), \
#                    epochs=1, verbose = 1)

    
    