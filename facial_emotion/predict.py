# Name: David Li
# Student Id: 101079263

import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

img_to_predict = 'man_sad.jpeg'

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
def emotion_analysis(emotion):
    # Create an array of indices for the emotions
    y_pos = np.arange(len(emotions))

    # Print the array of indices
    print(y_pos)

    # Plot a bar graph with emotion probabilities
    plt.bar(y_pos, emotion, align='center', alpha=0.5)

    # Add a label for the y-axis
    plt.ylabel('percentage')

    # Set the x-tick locations and labels to the emotions
    plt.xticks(y_pos, emotions)

    # Set the title of the graph
    plt.title('emotion')

    # Display the graph
    plt.show()

# Load an image and convert it to a numpy array
img = image.load_img(img_to_predict, grayscale=True, target_size=(48, 48))
x = image.img_to_array(img)

# Add a new dimension to the array to match the input shape of the model
x = np.expand_dims(x, axis=0)

# Normalize the array by dividing it by 255
x /= 255

# Load the model
model1 = load_model('my_model1.h5')

# Make a prediction using the loaded model
custom = model1.predict(x)

# Call the emotion_analysis function with the predicted emotions
emotion_analysis(custom[0])

# Convert the predicted emotions to a list
pre_emotions = list(custom[0])

# Print the predicted emotions
print(pre_emotions)

# Convert the numpy array to a float32 data type and reshape it to a 48x48 matrix
x = np.array(x, 'float32')
x = x.reshape([48, 48])

# Find the index of the maximum value in the predicted emotions list
idx = pre_emotions.index(max(pre_emotions))

# Set the title of the plot to the emotion with the maximum value
plt.title(emotions[idx])

# Display the plot in grayscale
plt.gray()

# Show the image in the plot
plt.imshow(x)

# Display the plot
plt.show()
