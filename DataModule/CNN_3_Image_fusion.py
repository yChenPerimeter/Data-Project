import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add
from tensorflow.keras.models import Model
from PIL import Image
import matplotlib.pyplot as plt

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = image.resize((256, 256))  # Resize image to 256x256
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Function to save the fused image
def save_image(image_array, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image_array = np.squeeze(image_array)  # Remove batch and channel dimensions
    image_array = (image_array * 255).astype(np.uint8)  # Convert back to [0, 255]
    Image.fromarray(image_array).save(output_path)

# Define the model
input1 = Input(shape=(256, 256, 1), name='input1')
input2 = Input(shape=(256, 256, 1), name='input2')
input3 = Input(shape=(256, 256, 1), name='input3')

conv1_1 = Conv2D(64, (3, 3), padding='same')(input1)
relu1_1 = ReLU()(conv1_1)
conv1_2 = Conv2D(64, (3, 3), padding='same')(relu1_1)
relu1_2 = ReLU()(conv1_2)
conv1_3 = Conv2D(64, (3, 3), padding='same')(relu1_2)
relu1_3 = ReLU()(conv1_3)

conv2_1 = Conv2D(64, (3, 3), padding='same')(input2)
relu2_1 = ReLU()(conv2_1)
conv2_2 = Conv2D(64, (3, 3), padding='same')(relu2_1)
relu2_2 = ReLU()(conv2_2)
conv2_3 = Conv2D(64, (3, 3), padding='same')(relu2_2)
relu2_3 = ReLU()(conv2_3)

conv3_1 = Conv2D(64, (3, 3), padding='same')(input3)
relu3_1 = ReLU()(conv3_1)
conv3_2 = Conv2D(64, (3, 3), padding='same')(relu3_1)
relu3_2 = ReLU()(conv3_2)
conv3_3 = Conv2D(64, (3, 3), padding='same')(relu3_2)
relu3_3 = ReLU()(conv3_3)

fused_features = Add()([relu1_3, relu2_3, relu3_3])

final_conv = Conv2D(1, (3, 3), padding='same')(fused_features)
output = ReLU()(final_conv)

model = Model(inputs=[input1, input2, input3], outputs=output)
model.compile(optimizer='adam', loss='mse')

# Load and preprocess the images
image_path1 = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /Capstone data /TrainingDataOG/Training/DCIS+/O21FK00004_P000012_S02_0233_1809_0076.png'
image_path2 = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /Capstone data /TrainingDataOG/Training/DCIS+/O21FK00004_P000012_S02_0208_1863_0076.png'
image_path3 = '/Users/ragini/Desktop/Perimeter Medical Imaging AI /My stuff /Capstone data /TrainingDataOG/Training/DCIS+/O21FK00004_P000012_S02_0208_1782_0076.png'
image1 = load_and_preprocess_image(image_path1)
image2 = load_and_preprocess_image(image_path2)
image3 = load_and_preprocess_image(image_path3)

# Combine images into batches for training
images1_batch = np.stack([image1, image1])
images2_batch = np.stack([image2, image2])
images3_batch = np.stack([image3, image3])

# Combine images for training
images1_batch = np.concatenate([images1_batch, images1_batch], axis=0)
images2_batch = np.concatenate([images2_batch, images2_batch], axis=0)
images3_batch = np.concatenate([images3_batch, images3_batch], axis=0)

# Assuming fused_images_batch is prepared correctly as needed for training
fused_images_batch = (images1_batch + images2_batch + images3_batch) / 3

# Train the model
model.compile(optimizer='adam', loss='mse')
model.fit([images1_batch, images2_batch, images3_batch], fused_images_batch, epochs=60, batch_size=2)

fused_image = model.predict([image1, image2, image3])
# Save the fused image
#output_path = '/Users/ragini/Desktop/Perimeter Medical Imaging AI/My stuff/Capstone data/Image fusion - result images/fused_image.png'
#save_image(fused_image, output_path)

# Display the fused image
fused_image_display = np.squeeze(fused_image)
plt.imshow(fused_image_display, cmap='gray')
plt.title('Fused Image')
plt.axis('off')
plt.show()
