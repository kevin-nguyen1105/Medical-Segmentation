from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def normalize(image):
    image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
    return image

def generate(X,Y):
    #Generator
    datagen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    X_datagen = ImageDataGenerator(**datagen_args)
    Y_datagen = ImageDataGenerator(**datagen_args)
    seed = 1
    X_datagen.fit(X, augment=True,seed=seed)
    Y_datagen.fit(Y, augment=True,seed=seed)
    X_generator = X_datagen.flow(X, batch_size=2, seed=seed)
    Y_generator = Y_datagen.flow(Y, batch_size=2, seed=seed)
    #Return generator
    train_generator = zip(X_generator, Y_generator)
    for (img,mask) in train_generator:
        img = normalize(img)
        mask = normalize(mask)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        yield (img,mask)