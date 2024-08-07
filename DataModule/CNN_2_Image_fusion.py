import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add
from PIL import Image
import numpy as np
import os
import os
from PIL import Image
import matplotlib.pyplot as plt


# Function to load and process images
def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = image.resize((256, 256))  # Resize image to 256x256
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image.reshape((256, 256, 1))  # Reshape to (256, 256, 1)

# Define the input layers
input1 = Input(shape=(256, 256, 1))
input2 = Input(shape=(256, 256, 1))

# Define the CNN layers
conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(input1)
conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(input2)

# Combine the features
combined = Add()([conv1, conv2])

# Define the output layer
output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(combined)

# Define the model
model = Model(inputs=[input1, input2], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse')


image1_path = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /Capstone data /TrainingDataOG/Training/DCIS+/O21FK00004_P000012_S02_0208_1755_0076.png'
image2_path = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /Capstone data /TrainingDataOG/Training/DCIS+/O21FK00004_P000012_S02_0208_1728_0076.png'


image1 = load_image(image1_path)
image2 = load_image(image2_path)

# Combine images for training
image1_batch = np.array([image1, image1])
image2_batch = np.array([image2, image2])
fused_image_batch = (image1_batch + image2_batch) / 2

# Train the model
model.fit([image1_batch, image2_batch], fused_image_batch, epochs=60, batch_size=2)

# Example usage to predict fusion
fused_image = model.predict([np.expand_dims(image1, axis=0), np.expand_dims(image2, axis=0)])
fused_image = (fused_image.squeeze() * 255).astype(np.uint8)
fused_image_pil = Image.fromarray(fused_image)


# Save the fused image

# Ensure the directory exists
#output_path = '/Users/ragini/Desktop/Perimeter Medical Imaging AI/My stuff/Capstone data/Image fusion - result images/fused_image.png'
#os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the fused image
#fused_image_pil = Image.fromarray(fused_image)
#fused_image_pil.save(output_path)

# Display the fused image
plt.imshow(fused_image, cmap='gray')
plt.title('Fused Image')
plt.axis('off')
plt.show()