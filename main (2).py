import os 
import cv2 
import sys
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
# import zlibwapi

dataset_path = 'C:/Users/Desktop/currency-detection/Dataset'
# # class_folders = os.listdir(dataset_path)
# # num_classes = len(class_folders)

# # for class_index, class_folder in enumerate(class_folders):
# #     x=class_folder
# #     x = x.split(' ')[0]
# #     y = x+'_hk'
# #     print(x)
# #     class_path = os.path.join(dataset_path, class_folder)
# #     new =  os.path.join(dataset_path, y)
# #     if os.path.isdir(class_path):
# #         os.rename(class_path, y)


# target_size = (224,224)

# batch_size = 3

# datagen = ImageDataGenerator(
#     rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
#     rotation_range=20,    # Randomly rotate images
#     width_shift_range=0.2,  # Randomly shift images horizontally
#     height_shift_range=0.2,  # Randomly shift images vertically
#     shear_range=0.2,        # Apply shear transformation
#     zoom_range=0.2,         # Randomly zoom in and out
#     horizontal_flip=True,   # Randomly flip images horizontally
#     fill_mode='nearest'     # Fill missing pixels with the nearest available pix ccel
# )

# # train_generator = ImageDataGenerator.flow_from_directory(
# #     dataset_path,
# #     target_size=target_size,
# #     batch_size=batch_size,
# #     class_mode='categorical',
# #     shuffle=True
# # )


# # validation_generator = ImageDataGenerator.flow_from_directory(
# #     dataset_path,
# #     target_size=target_size,
# #     batch_size=batch_size,
# #     class_mode='categorical',
# #     shuffle=False
# # )


# def preprocess_image(image_path):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, target_size)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32)
#     img = img/255.0
#     return img 

# X= []
# y= []

# class_folders = os.listdir(dataset_path)
# num_classes = len(class_folders)

# for class_index, class_folder in enumerate(class_folders):
#     print(class_folder)
#     class_path = os.path.join(dataset_path, class_folder)
#     if os.path.isdir(class_path):
#         for filename in os.listdir(class_path):
#             if filename.endswith('.jpg') or filename.endswith('.png'):
#                 image_path = os.path.join(class_path, filename)
#                 img = preprocess_image(image_path)
#                 X.append(img)
#                 y.append(class_index)

# X = np.array(X)
# y = np.array(y)
# print(y)

# # split dataset into train and test

# from sklearn.model_selection import train_test_split

# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# from keras.utils import to_categorical


# y_train_encoded = to_categorical(y_train, num_classes=num_classes)
# y_val_encoded = to_categorical(y_val, num_classes=num_classes)
# y_test_encoded = to_categorical(y_test, num_classes=num_classes)

# import tensorflow as tf
# from keras.models import Model
# from keras.layers import Input, Dense, GlobalAveragePooling2D
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, EarlyStopping

# base_model = tf.keras.applications.MobileNetV2(
#     input_shape=(224,224,3),
#     include_top=False,
#     weights='imagenet'
# )

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x= Dense(1024, activation='relu')(x)
# predictions = Dense(num_classes, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# for layer in base_model.layers:
#     layer.trainable = False

# model.compile(
#     optimizer=Adam(lr=0.001),
#     loss = 'categorical_crossentropy',
#     metrics=['accuracy']
# )

# checkpoint = ModelCheckpoint('currency_detection_model.h5', save_best_only=True)
# early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# epochs = 8


# result = model.fit(
#     X_train,
#     y_train_encoded,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(X_val, y_val_encoded),
#     callbacks=[checkpoint, early_stopping]
# )

# test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
# print(f'test loss is: {test_loss}; test accuracy is {test_accuracy}')

# model.save('final_currency_detection_model.h5')



from keras.models import load_model

# Load the .h5 model file
model = load_model('final_currency_detection_model.h5')
import numpy as np
from PIL import Image

# Load and preprocess the image (ensure it matches the model's input size and preprocessing)
img = Image.open('100taka.jpg')
img = img.resize((224, 224))  # Adjust the size according to your model's input size
img = np.array(img) / 255.0  # Normalize pixel values (if needed)

# Make a prediction
predictions = model.predict(np.expand_dims(img, axis=0))

# Get the predicted class (for classification tasks)
predicted_class = np.argmax(predictions)
matrix = ['1000_bd', '1000_hk', '1000_pk', '100_bd', '100_hk','100_in','100_pk','10_bd','10_hk','10_in','10_pk','2000_in','200_bd','200_in',"20_bd","20_hk","20_in","20_pk","5000_pk","500_bd","500_hk","500_in","500_pk","50_bd","50_hk","50_in","50_pk","5_bd"]
predicted_class = matrix[predicted_class]

print(f"Predicted Class: {predicted_class}")