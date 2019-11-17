import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from network import Unet
from generate_data import generate

IMAGE_SHAPE_3D = (512, 512, 1)
IMAGE_SHAPE = (512, 512)
LEARNING_RATE = 1e-4
STEPS_PER_EPOCH = 5
EPOCHS = 5

#Normalize the pixels to range 0-1
def normalize(image):
    image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
    return image

#Convert original images to size (512, 512) graysacle image
def read_images(folder):
    images = []
    for image in sorted(os.listdir(folder)):
        image_np = cv2.imread(folder + image)
        image_resized = cv2.resize(image_np, IMAGE_SHAPE)
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        gray = normalize(gray)
        gray = np.expand_dims(gray,axis=2)
        images.append(gray)
    return np.array(images)

#Read the masks files and produce the corresding original images + masks
def read_masks(folder, originalImages):
	originalTrain = []
	maskTrain = []
	mask_files = sorted(os.listdir(folder))
	for mask_file in mask_files:
		#Get the corresponding original image
		imageNo = int(mask_file.split("_")[1]) - 1
		originalTrain.append(originalImages[imageNo])
		#Process the mask
		mask_np = cv2.imread(folder + mask_file)
		mask_resized = cv2.resize(mask_np, IMAGE_SHAPE)
		gray = cv2.cvtColor(mask_resized, cv2.COLOR_BGR2GRAY)
		gray = normalize(gray)
		gray[gray >= 0.5] = 1
		gray[gray < 0.5] = 0
		gray = np.expand_dims(gray,axis=2)
		maskTrain.append(gray)
	return np.array(originalTrain), np.array(maskTrain)

#Convert masks to 0/1 values
def process_masks(masks):
    for i in range(masks.shape[0]):
        masks[i] = normalize(masks[i])
        masks[i][masks[i] >= 0.5] = 1
        masks[i][masks[i] < 0.5] = 0
    return masks

#Write images to folder
def write_images(images, result_folder):
    count = 1
    for image in images:
        #Process
        image = normalize(image)
        image = image * 255
        #Write to file
        cv2.imwrite(result_folder + str(count).zfill(2) + '.jpg', image.squeeze())
        count += 1

#Create loss plot from keras history
def create_plots(history, maskType):
	#Loss plot
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title(maskType + " Loss")
	plt.ylabel('Dice Coef Loss')
	plt.xlabel('Epochs')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig("Plots/Loss/" + "".join(maskType.split()) + ".png")
	plt.figure()
	#Accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title(maskType + ' Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig("Plots/Accuracy/" + "".join(maskType.split()) + ".png")
	plt.figure()
	#Dice Coef
	plt.plot(history.history['dice_coef'])
	plt.plot(history.history['val_dice_coef'])
	plt.title(maskType + ' Dice Coef')
	plt.ylabel('Dice Coef')
	plt.xlabel('Epochs')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig("Plots/DiceCoef/" + "".join(maskType.split()) + ".png")
	plt.figure()
	
def main():
    #Initialize U-net model
    print("Initializing Networks...")
    haemorrhagesModel = Unet(LEARNING_RATE, IMAGE_SHAPE_3D).get_model()
    hardExudatesModel = Unet(LEARNING_RATE, IMAGE_SHAPE_3D).get_model()
    microaneurysmsModel = Unet(LEARNING_RATE, IMAGE_SHAPE_3D).get_model()
    softExudatesModel = Unet(LEARNING_RATE, IMAGE_SHAPE_3D).get_model()

    #Original images
    print("Reading original images...")
    originalTrain = read_images("Train/original_retinal_images/")
    originalTest = read_images("Test/original_retinal_images/")
    originalImages = np.concatenate((originalTrain, originalTest), axis=0)

    #Train masks
    print("Reading train masks...")
    originalHaemorrhagesTrain, haemorrhagesTrain = read_masks("Train/masks_Haemorrhages/", originalImages)
    originalHardExudatesTrain, hardExudatesTrain = read_masks("Train/masks_Hard_Exudates/", originalImages)
    originalMicroaneurysmsTrain, microaneurysmsTrain = read_masks("Train/masks_Microaneurysms/", originalImages)
    originalSoftExudatesTrain, softExudatesTrain = read_masks("Train/masks_Soft_Exudates/", originalImages)

    #Test masks
    print("Reading test masks...")
    originalHaemorrhagesTest, haemorrhagesTest = read_masks("Test/masks_Haemorrhages/", originalImages)
    originalHardExudatesTest, hardExudatesTest = read_masks("Test/masks_Hard_Exudates/", originalImages)
    originalMicroaneurysmsTest, microaneurysmsTest = read_masks("Test/masks_Microaneurysms/", originalImages)
    originalSoftExudatesTest, softExudatesTest = read_masks("Test/masks_Soft_Exudates/", originalImages)

    #Image Generators
    haemorrhagesGen = generate(originalHaemorrhagesTrain, haemorrhagesTrain)
    hardExudatesGen = generate(originalHardExudatesTrain, hardExudatesTrain)
    microaneurysmsGen = generate(originalMicroaneurysmsTrain, microaneurysmsTrain)
    softExudatesGen = generate(originalSoftExudatesTrain, softExudatesTrain)

    #Train on generated data
    print("Start Training!")
    haemorrhagesHistory = haemorrhagesModel.fit_generator(haemorrhagesGen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=(originalHaemorrhagesTest, haemorrhagesTest))
    hardExudatesHistory = hardExudatesModel.fit_generator(hardExudatesGen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=(originalHardExudatesTest, hardExudatesTest))
    microaneurysmsHistory = microaneurysmsModel.fit_generator(microaneurysmsGen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=(originalMicroaneurysmsTest, microaneurysmsTest))
    softExudatesHistory = softExudatesModel.fit_generator(softExudatesGen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=(originalSoftExudatesTest, softExudatesTest))

    #Save models
    print("Saving Models...")
    haemorrhagesModel.save("Models/haemorrhages.h5")
    hardExudatesModel.save("Models/hardExudates.h5")
    microaneurysmsModel.save("Models/microaneurysms.h5")
    softExudatesModel.save("Models/softExudates.h5")

    #Create loss and metric plots
    print("Creating Plots...")
    create_plots(haemorrhagesHistory, "Haemorrhages")
    create_plots(hardExudatesHistory, "Hard Exudates")
    create_plots(microaneurysmsHistory, "Microaneurysms")
    create_plots(softExudatesHistory, "Soft Exudates")


    #Produce results
    print("Producing results...")
    haemorrhagesResults = process_masks(haemorrhagesModel.predict(originalHaemorrhagesTest))
    write_images(haemorrhagesResults, "Result/masks_Haemorrhages/")
    hardExudatesResults = process_masks(hardExudatesModel.predict(originalHardExudatesTest))
    write_images(haemorrhagesResults, "Result/masks_Hard_Exudates/")
    microaneurysmsResults = process_masks(microaneurysmsModel.predict(originalMicroaneurysmsTest))
    write_images(haemorrhagesResults, "Result/masks_Microaneurysms/")
    softExudatesResults = process_masks(softExudatesModel.predict(originalSoftExudatesTest))
    write_images(haemorrhagesResults, "Result/masks_Soft_Exudates/")

if __name__ == "__main__":
    main()
