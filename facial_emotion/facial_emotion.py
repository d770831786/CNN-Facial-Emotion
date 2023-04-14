# Name: David Li
# Student Id: 101079263

import pandas as pd #pandas 1.2.2
import numpy as np#numpy 1.21.0
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D,Conv2D,BatchNormalization,MaxPooling2D
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.utils import to_categorical

# using panda to read the csv file
#data.head(10) read the first 10 rows of the csv file to see the format of the .csv file
data = pd.read_csv('icml_face_data.csv')

# Extract image
def prepare_data(data):
    # Extract pixel data and convert to numpy array
    image_array = np.array([np.fromstring(pixel, sep=' ').astype(int).reshape(48, 48)
                            for pixel in data[' pixels']])

    # Extract labels and convert to numpy array
    image_label = data['emotion'].values.astype(int)

    return image_array, image_label


#data[' Usage'].value_counts()  count the number of Training dataset and Test dataset
#Training       28709
#PrivateTest     3589
#PublicTest      3589
#Name:  Usage, dtype: int64

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
#devide the dataset into training dataset, valid dataset and test dataset
train_image_array, train_image_label = prepare_data(data[data[' Usage']=='Training'])
val_image_array, val_image_label = prepare_data(data[data[' Usage']=='PrivateTest'])
test_image_array, test_image_label = prepare_data(data[data[' Usage']=='PublicTest'])

# Preprocessing
# Convert and normalise integers to floating point numbers
# Reshape the numpy array of training images to have a shape of (number of images, height, width, channels)
train_images = train_image_array.reshape((train_image_array.shape[0], 48, 48, 1))
train_images = train_images.astype('float32')/255
val_images = val_image_array.reshape((val_image_array.shape[0], 48, 48, 1))
val_images = val_images.astype('float32')/255
test_images = test_image_array.reshape((test_image_array.shape[0], 48, 48, 1))
test_images = test_images.astype('float32')/255

# Get the labels of images
train_labels = to_categorical(train_image_label)
val_labels = to_categorical(val_image_label)
test_labels = to_categorical(test_image_label)

# calculate the weights of different type of emotions
class_weight = dict(zip(range(0, 7), (((data[data[' Usage']=='Training']['emotion'].value_counts()).sort_index())/len(data[data[' Usage']=='Training']['emotion'])).tolist()))

# Creating a sequential model
model = models.Sequential()
# Adding a convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
# Adding batch normalization to improve training efficiency
model.add(BatchNormalization())
# Adding a max pooling layer with 2x2 pool size
model.add(MaxPooling2D((2, 2)))
# Adding another convolutional layer with 64 filters and a 3x3 kernel
model.add(Conv2D(64, (3, 3), activation='relu'))
# Adding batch normalization
model.add(BatchNormalization())
# Adding another max pooling layer
model.add(MaxPooling2D((2, 2)))
# Adding another convolutional layer with 128 filters and a 3x3 kernel
model.add(Conv2D(128, (3, 3), activation='relu'))
# Adding batch normalization
model.add(BatchNormalization())
# Adding another max pooling layer
model.add(MaxPooling2D((2, 2)))
# Flattening the output of the previous layer
model.add(Flatten())
# Adding a dropout layer with 50% probability of dropping units
model.add(Dropout(0.5))
# Adding a fully connected (dense) layer with 256 neurons and ReLU activation
model.add(Dense(256, activation='relu'))
# Adding batch normalization
model.add(BatchNormalization())
# Adding another dropout layer
model.add(Dropout(0.5))
# Adding another fully connected (dense) layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))
# Adding batch normalization
model.add(BatchNormalization())
# Adding the output layer with 7 neurons (one for each emotion) and softmax activation
model.add(Dense(7, activation='softmax'))

# Compiling the model with Adam optimizer, learning rate of 1e-3, categorical crossentropy loss, and accuracy metric
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Printing the summary of the model
model.summary()

# Train the model
history = model.fit(train_images, train_labels,
                    validation_data=(val_images, val_labels),
                    class_weight = class_weight,
                    epochs=16,
                    batch_size=256)

# Evaluate the model using test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test caccuracy:', test_acc)

# Predict the emotion of test images
pred_test_labels = model.predict(test_images)
model.save('my_model1.h5')


################################ Analysis of the model ################################

#draw images of specific emotion from dataset, ex label=0 means draw the angry emotion
def plot_examples(label=0):
    # Create a figure with 1 row and 5 columns, with the size of 25x12 inches
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25, 12))
    # Adjust the spacing between subplots
    fig.subplots_adjust(hspace=.2, wspace=.2)
    # Flatten the 5 subplots into a 1D array for easier manipulation
    axs = axs.flatten()
    # Select the first 5 images with the given emotion label from the dataset
    for i, idx in enumerate(data[data['emotion'] == label].index[:5]):
        # Display the image in the i-th subplot
        axs[i].imshow(train_images[idx][:, :, 0], cmap='gray')
        # Set the title of the i-th subplot to the corresponding emotion category
        axs[i].set_title(emotions[train_labels[idx].argmax()])
        # Hide the x-axis tick labels of the i-th subplot
        axs[i].set_xticklabels([])
        # Hide the y-axis tick labels of the i-th subplot
        axs[i].set_yticklabels([])


# Define a function to plot images corresponding to each emotion
def plot_all_emotions():
    # Create a figure with 1 row and 7 columns and set the figure size
    fig, axs = plt.subplots(1, 7, figsize=(30, 12))

    # Adjust the spacing between the subplots
    fig.subplots_adjust(hspace=.2, wspace=.2)

    # Flatten the axes array
    axs = axs.ravel()

    # Loop through each emotion
    for i in range(7):
        # Get the index of the first image for the current emotion
        idx = train_labels[:, i].nonzero()[0][0]

        # Display the image for the current emotion
        axs[i].imshow(train_images[idx], cmap='gray')

        # Set the title for the current subplot
        axs[i].set_title(emotions[i])

        # Remove the x and y axis labels for the current subplot
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])


# Function to plot the image and compare the prediction results with the label
def plot_image_and_emotion(test_image_array, test_image_label, pred_test_labels, image_number):

    # Create a figure with 1 row and 2 columns, with the size of 12x6 inches
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    # Get the emotion labels for the bar chart
    bar_label = emotions.values()

    # Show the test image in the first subplot
    axs[0].imshow(test_image_array[image_number], 'gray')
    # Set the title of the first subplot to the emotion label of the test image
    axs[0].set_title(emotions[test_image_label[image_number]])

    # Create a bar chart in the second subplot, with the predicted labels as the data
    axs[1].bar(bar_label, pred_test_labels[image_number], color='orange', alpha=0.7)
    # Add a grid to the second subplot
    axs[1].grid()

    # Show the figure
    plt.show()


def plot_compare_distributions(array1, array2, title1='', title2=''):
    # Convert the max index of each array to the corresponding emotion label and create two dataframes
    df_array1 = pd.DataFrame({'emotion': array1.argmax(axis=1)})
    df_array2 = pd.DataFrame({'emotion': array2.argmax(axis=1)})

    # Create a figure with 1 row and 2 columns, with the size of 12x6 inches
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    # Get the emotion labels as x values for the bar charts
    x = list(emotions.values())

    # Plot the distribution of array1 in the first subplot
    y = df_array1['emotion'].value_counts().sort_index().reindex(emotions.keys(), fill_value=0)
    axs[0].bar(x, y, color='blue')
    axs[0].set_title(title1)
    axs[0].grid()

    # Plot the distribution of array2 in the second subplot
    y = df_array2['emotion'].value_counts().sort_index().reindex(emotions.keys(), fill_value=0)
    axs[1].bar(x, y)
    axs[1].set_title(title2)
    axs[1].grid()

    # Show the figure
    plt.show()


# Draw the loss_train and loss_val curve during the training process
loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'bo', label='loss_train')
plt.plot(epochs, loss_val, 'b', label='loss_val')
plt.title('value of the loss function')
plt.xlabel('epochs')
plt.ylabel('value of the loss function')
plt.legend()
plt.grid()
plt.show()

# Draw the accuracy_train and accuracy_val curve during the training process
acc = history.history['accuracy']
acc_val = history.history['val_accuracy']
epochs = range(1, len(loss)+1)
plt.plot(epochs, acc, 'bo', label='accuracy_train')
plt.plot(epochs, acc_val, 'b', label='accuracy_val')
plt.title('accuracy')
plt.xlabel('epochs')
plt.ylabel('value of accuracy')
plt.legend()
plt.grid()
plt.show()

# Draw the No.248 of dataset and predict its emotion
plot_image_and_emotion(test_image_array, test_image_label, pred_test_labels, 248)
# Draw and compare the distribution graph of test labels and predict labels
plot_compare_distributions(test_labels, pred_test_labels, title1='test labels', title2='predict labels')

# Draw the confusion matrix
df_compare = pd.DataFrame()
df_compare['real'] = test_labels.argmax(axis=1)
df_compare['pred'] = pred_test_labels.argmax(axis=1)
df_compare['wrong'] = np.where(df_compare['real']!=df_compare['pred'], 1, 0)

conf_mat = confusion_matrix(test_labels.argmax(axis=1), pred_test_labels.argmax(axis=1))

fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
                                show_normed=True,
                                show_absolute=False,
                                class_names=emotions.values(),
                                figsize=(8, 8))
fig.show()
