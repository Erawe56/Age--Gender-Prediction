import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import load_img
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from PIL import Image

warnings.filterwarnings('ignore')

# Directory containing the dataset
BASE_DIR = 'D:/vg/data/UTKFace'

# Labels: age, gender, ethnicity
image_paths = []
age_labels = []
gender_labels = []

for filename in tqdm(os.listdir(BASE_DIR)):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)

# Convert to dataframe
df = pd.DataFrame({
    'image': image_paths,
    'age': age_labels,
    'gender': gender_labels
})

# Map labels for gender
gender_dict = {0: 'Male', 1: 'Female'}

# Display an example image
img = Image.open(df['image'][0])
plt.axis('off')
plt.imshow(img)
plt.show()

# Display age distribution
sns.histplot(df['age'], kde=True)
plt.show()

# Display gender count
sns.countplot(x=df['gender'], palette="pastel")
plt.xticks(ticks=[0, 1], labels=["Male", "Female"])
plt.show()

# Display a grid of images
plt.figure(figsize=(20, 20))
files = df.iloc[0:25]

for index, file, age, gender in files.itertuples():
    plt.subplot(5, 5, index + 1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title(f"Age: {age} Gender: {gender_dict[gender]}")
    plt.axis('off')
plt.show()

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')
        img = img.resize((128, 128), Image.LANCZOS)
        img = np.array(img)
        features.append(img)
        
    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features

X = extract_features(df['image'])
X = X / 255.0  # Normalize the images

y_gender = np.array(df['gender'])
y_age = np.array(df['age'])

# Normalize the age values
y_age = y_age / 100.0

# Define input shape
input_shape = (128, 128, 1)

inputs = Input(shape=input_shape)
# Convolutional layers
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu')(maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

flatten = Flatten()(maxp_4)

# Fully connected layers
dense_1 = Dense(256, activation='relu')(flatten)
dropout_1 = Dropout(0.4)(dense_1)
output_1 = Dense(1, activation='sigmoid', name='gender_out')(dropout_1)

dense_2 = Dense(256, activation='relu')(flatten)
dropout_2 = Dropout(0.4)(dense_2)
output_2 = Dense(1, activation='linear', name='age_out')(dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(
    loss=['binary_crossentropy', 'mean_absolute_error'],
    optimizer="adam",
    metrics={'gender_out': 'accuracy', 'age_out': 'mean_absolute_error'}
)

# Train model
history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=30, validation_split=0.2)

# Save model
model.save("D:/vg/agegender_resaved.h5")
print("Model saved successfully.")



