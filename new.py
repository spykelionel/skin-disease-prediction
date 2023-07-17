# Import required moduls/libs for our model
import zipfile
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from tensorflow.keras.layers import BatchNormalization
import itertools
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
# used for converting labels to one-hot-encoding
from keras.utils.np_utils import to_categorical
import numpy as np          # linear algebra
import pandas as pd         # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dropout, Activation
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Flatten, \
    Dense, Input, Activation, Dropout, GlobalAveragePooling2D, AveragePooling2D
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import cv2
from cv2 import imread, resize  # manipulating the images
from tensorflow.keras.optimizers import Adam
import os

# Reading the meta date of our data frame.
df_skin = pd.read_csv('HAM10000_metadata_new.csv')
# Display the first 10 lines
df_skin.head(10)

# Lesion/dis names are given in the description of the data set.
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma',
    'ns': 'Normal Skin'
}

lesion_ID_dict = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6,
    'ns': 7
}

# Lesion and it's abbriv.
lesion_names = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions ',
                'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions',
                'Dermatofibroma', 'Normal Skin']
lesion_names_short = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df', 'ns']

# Maping the lesion type and ID to a dict.
df_skin['lesion_type'] = df_skin['dx'].map(lesion_type_dict)
df_skin['lesion_ID'] = df_skin['dx'].map(lesion_ID_dict)

# Display the total found images.
print('Total number of images', len(df_skin))
print('The problem is unbalanced, since Melanocytic nevi is much more frequent that other labels')

# Display the count of each lesion.
df_skin['lesion_type'].value_counts()

# Reading a random image from our data set
fname_images = np.array(df_skin['image_id'])
file_to_read = 'HAM10000_images_part_1/'+str(fname_images[13])+'.jpg'

# Resizing the read image to 100x100
img = imread(file_to_read)
img2 = resize(img, (100, 100))

# Show one exampe image before and after


def show_single_image():
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img[:, :, ::-1])
    plt.title('Original image')
    plt.subplot(1, 2, 2)
    plt.imshow(img2[:, :, ::-1])
    plt.title('Resized image for DenseNet')
    plt.show()


def produce_new_img(img2: cv2) -> tuple:
    """
    function to reproduse a new manipulated (rotating of flipping the original one)
    image from the read one, To increase the dimension of the dataset, avoiding overfitting of a single class.

    Args:
        img2 (cv2): the read image from cv2 module.

    Returns:
        new_images (tuple): a tuple of the new manipulated images.
    """
    imga = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    imgb = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    imgc = cv2.rotate(img2, cv2.ROTATE_180)
    imgd = cv2.flip(img2, 0)
    imge = cv2.flip(img2, 1)
    new_imges = imga, imgb, imgc, imgd, imge
    return new_imges


def show_example() -> None:
    """
    Display an image after manipulating it in produce_new_img() function
    """
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(img2[:, :, ::-1])
    for i in range(5):
        plt.subplot(2, 3, 2+i)
        plt.imshow(new_img[i][:, :, ::-1])
    plt.tight_layout()
    plt.show()


# Invoking our function :)
new_img = produce_new_img(img2)
# Display an example by invoking show_example func
# show_example()

x = []          # Hold resized images.
y = []          # Hold image lesion ID from the data set.

# Listing all files in the part_1, part_2 dirs
lista1 = os.listdir('HAM10000_images_part_1/')
lista2 = os.listdir('HAM10000_images_part_2/')
lista3 = os.listdir('Normal_Skin/')

def get_resized_image(file_to_read):
    img = imread(file_to_read)
    img = resize(img, (100, 100))
    return img

# [+] Handling images from part 1 directory
for i in range(len(lista1)):
    # [+] Features: reading and resize the photo.
    fname_image = lista1[i]
    fname_ID = fname_image.replace('.jpg', '')
    file_to_read = 'HAM10000_images_part_1/' + \
        str(fname_image)  # resolve image name
    # read the image
    img = get_resized_image(file_to_read)
    # append the new image to the list x.
    x.append(img)

    # Targets: Finding the image lesion ID and append it to the y list.
    output = np.array(df_skin[df_skin['image_id'] == fname_ID].lesion_ID)
    y.append(output[0])
    # print(output[0])

    # add more images for class between 1-6, rotating them
    if output != 0:
        new_img = produce_new_img(img2)
        for i in range(5):
            x.append(new_img[i])
            y.append(output[0])

# [+] Handling images from part 2 directory
for i in range(len(lista2)):

    # [+] Features: reading and resize the photo.
    fname_image = lista2[i]
    fname_ID = fname_image.replace('.jpg', '')
    file_to_read = 'HAM10000_images_part_2/' + str(fname_image)

    img = get_resized_image(file_to_read)
    x.append(img)

    # Targets: Finding the image lesion ID and append it to the y list.
    output = np.array(df_skin[df_skin['image_id'] == fname_ID].lesion_ID)
    # print("output[0]:",output[0])
    # print("y:",y)
    y.append(output[0])

    # [+] Add more images for class between 1-6
    if output != 0:
        new_img = produce_new_img(img2)
        for i in range(5):
            x.append(new_img[i])
            y.append(output[0])

# [+] Handling images from part 3 directory, only normal skin here
for i, image in enumerate(lista3):

    # [+] Features: reading and resize the photo.
    fname_image = lista3[i]
    fname_ID = fname_image.replace('.jpg', '')
    file_to_read = 'Normal_Skin/' + str(image)

    img = get_resized_image(file_to_read)
    x.append(img)
    output = 7
    # Targets: Finding the image lesion ID and append it to the y list.
    output = np.array(df_skin[df_skin['image_id'] == fname_ID].lesion_ID) # output = [['lession_id']]
    y.append(output[0])

    if output != 0:
        new_img = produce_new_img(img)
        for i in range(5):
            x.append(new_img[i])
            y.append(output[0])

    # # [+] Inform the user with the number of loaded images each 100 img.
    # if i % 100 == 0:
    #     print(len(lista3) + i, 'images loaded')

# add one more class
x = np.array(x)
y = np.array(y)

# Filter values equal to the class 7 from the lession_id y.
# filtered_y = y[y == 7]
# print("Length of class 7", filtered_y)
# print("x:", x[:5])
# print("y:", x[:5])

# convert y (targets) array as required by softmax activation function
y_train = to_categorical(y, num_classes=8)

# split in 80% training and 20% test data
X_train, X_test, y_train, y_test = train_test_split(x,                  # Images array.
                                                    # The training set.
                                                    y_train,
                                                    # Split data set into 20/80.
                                                    test_size=0.20,
                                                    # Shuffling number to random the set.
                                                    random_state=50,
                                                    stratify=y)       # Mix training and test sets.
# [+] Display the count of train/test data set.
print('Train dataset shape', X_train.shape)
print('Test dataset shape', X_test.shape)


def show_neg_figuers() -> None:
    """ Display negative figuers of the classes. """
    # Figure, Axes
    _, ax = plt.subplots(1, 7, figsize=(30, 30))
    for i in range(7):
        # set the image to negative.
        ax[i].set_axis_off()
        # Display the img.
        ax[i].imshow(X_train[i])
        # Set image title.
        ax[i].set_title(lesion_names[np.argmax(y_train[i])])


show_neg_figuers()


def est_class_weights(dis_id: np.array) -> dict:
    """Estimate class weights for unbalanced datasets.

    Args:
        dis_id (np.array): numpy array of dis IDs

    Returns:
        dict: Estimated class weights for for unbalanced datasets.
    """
    class_weights = np.around(compute_class_weight(
        class_weight='balanced', classes=np.unique(dis_id), y=y), 2)
    class_weights = dict(zip(np.unique(dis_id), class_weights))


# Append class 7 rows with value to the 'lesion_id' column
# new_rows = pd.Series(filtered_y)
# df_skin = df_skin.append(new_rows, ignore_index=True)

y_id = np.array(df_skin['lesion_ID'])

print("df_skin['lesion_ID']:", df_skin['lesion_ID'])
new_class_weights = est_class_weights(y_id)
print('The problem is unbalanced. We need to provide class-weights')
print(new_class_weights)


model = Sequential()


def model_arch():
    model.add(Conv2D(filters=96,
                     kernel_size=(11, 11),
                     strides=(4, 4),
                     activation='relu',
                     input_shape=(100, 100, 3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(filters=256,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     activation='relu',
                     padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Conv2D(filters=384,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     activation='relu',
                     padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=384,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     activation='relu',
                     padding="same"))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     activation='relu',
                     padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())  # [+] Convert the Conv2D objects into one List.

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # [+] 7th Dense layer
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # [+] 8th output layer
    model.add(Dense(7, activation='softmax'))


model_arch()


def mod_checkpoint_callback() -> None:
    """
    Saving our model

    Returns:
        None: Saving a checkpoint of the model.
    """
    trained_model = ModelCheckpoint(filepath='model.h5',  # result file name
                                    # Save all training results/params.
                                    save_weights_only=False,
                                    # check our model accuracy if it's step forward.
                                    monitor='val_accuracy',
                                    # enable auto save.
                                    mode='auto',
                                    save_best_only=True,         # if ac_new > ac_old
                                    verbose=1)
    return trained_model


# Montoring the training procces in each epoch.
early_stopping_monitor = EarlyStopping(patience=100, monitor='val_accuracy')

model_checkpoint_callback = mod_checkpoint_callback()

# Estimate the model data if it was big one.
optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-3)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    zoom_range=0.2, horizontal_flip=True, shear_range=0.2)
datagen.fit(X_train)

batch_size = 32     # samples in the network at once.
epochs = 100        # epochs number.


# org model result data
history = model.fit(datagen.flow(X_train, y_train),
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    callbacks=[early_stopping_monitor,
                               model_checkpoint_callback],
                    validation_data=(X_test, y_test),
                    class_weight=new_class_weights
                    )

# [+] inform the user with model Accuracy %
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1] * 100))

# Save the nodel as a tensor model
print("Saving the model")
model.save("/kaggle/working/saved_models/v1")
print("Saving the model")

# zip the model so user can download


def zipdir(path, ziph):
    # Iterate over all the files in the directory
    for root, dirs, files in os.walk(path):
        for file in files:
            # Create the full file path by joining the root and file name
            file_path = os.path.join(root, file)
            # Add the file to the zip archive
            ziph.write(file_path)


# Set the name of the zip archive
zip_file_name = 'latest2.zip'
# Set the path of the directory to zip
dir_to_zip = '/kaggle/working/saved_models/v2'
# Create the zip archive
print("Begin, zip directory")
with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as ziph:
    zipdir(dir_to_zip, ziph)
    print("Done zipping directory")


def display_accuracy() -> None:
    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
    plt.show()


def display_loss() -> None:
    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


y_pred = model.predict(X_test)


def test_model() -> tuple:
    """ Tunning the accurate results and inaccurate results

    Returns:
        (total, accurate) [tuple]: tuple of total tested test-cases, accurate
    """
    total = 0
    accurate = 0
    accurateindex = []
    wrongindex = []
    for i in range(len(y_pred)):
        if np.argmax(y_pred[i]) == np.argmax(y_test[i]):
            accurate += 1
            accurateindex.append(i)
        else:
            wrongindex.append(i)
        total += 1
    return (total, accurate)


total, accurate = test_model()
print('Total-test-data;', total, '\taccurately-predicted-data:',
      accurate, '\t wrongly-predicted-data: ', total - accurate)
print('Accuracy:', round(accurate / total * 100, 3), '%')


best_model = load_model('./model.h5')

# Compute predictions
y_pred_prob = np.around(best_model.predict(X_test), 3)
y_pred = np.argmax(y_pred_prob, axis=1)
y_test2 = np.argmax(y_test, axis=1)

scores = best_model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1] * 100))

plt.figure(figsize=(16, 16))
for i in range(16):
    plt.subplot(4, 4, i+1)
    index = i+100
    plt.imshow(X_test[index, :, :, ::-1])
    label_exp = lesion_names[y_test2[index]]  # expected label
    label_pred = lesion_names[y_pred[index]]  # predicted label
    label_pred_prob = round(np.max(y_pred_prob[index])*100)
    plt.title('Expected:'+str(label_exp)+'\n Pred.:' +
              str(label_pred)+' ('+str(label_pred_prob)+'%)')
plt.ylabel('')
plt.tight_layout()
plt.savefig('final_figure.png', dpi=300)
plt.show()

acc_tot = []

for i in range(7):
    acc_parz = round(np.mean(y_test2[y_test2 == i] == y_pred[y_test2 == i]), 2)
    lab_parz = lesion_names[i]
    print('accuracy for', lab_parz, '=', acc_parz)
    acc_tot.append(acc_parz)

acc_tot = np.array(acc_tot)
freq = np.unique(y_test2, return_counts=True)[1]

np.sum(acc_tot*freq)/np.sum(freq)

print("Saving the model as TF")
model.save("saved_models/v1")
print("Saved the model")
