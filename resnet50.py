# -*- coding: utf-8 -*-
"""ResNet50.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KwQedtonua9iHJeSRgD7zZzau-GHoMen
"""

#Importing libraries
import os
import seaborn as sns
import pandas as pd
import numpy as np
import cv2 as cv
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from google.colab import drive
drive.mount('/content/gdrive')

dataset = pd.read_csv("/content/gdrive/MyDrive/full_df.csv")

dataset

def process_dataset(data):

    data["left_cataract"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("cataract",x))
    data["right_cataract"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("cataract",x))

    data["LD"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("non proliferative retinopathy",x))
    data["RD"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("non proliferative retinopathy",x))

    data["LG"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("glaucoma",x))
    data["RG"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("glaucoma",x))

    data["LH"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("hypertensive",x))
    data["RH"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("hypertensive",x))

    data["LM"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("myopia",x))
    data["RM"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("myopia",x))

    data["LA"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("macular degeneration",x))
    data["RA"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("macular degeneration",x))

    data["LO"] = data["Left-Diagnostic Keywords"].apply(lambda x: has_condn("drusen",x))
    data["RO"] = data["Right-Diagnostic Keywords"].apply(lambda x: has_condn("drusen",x))

    left_cataract_images = data.loc[(data.C ==1) & (data.left_cataract == 1)]["Left-Fundus"].values
    right_cataract_images = data.loc[(data.C == 1) & (data.right_cataract == 1)]["Right-Fundus"].values

    left_normal = data.loc[(data.C == 0) & (data["Left-Diagnostic Keywords"] == "normal fundus")]['Left-Fundus'].sample(350,random_state=42).values
    right_normal = data.loc[(data.C == 0) & (data["Right-Diagnostic Keywords"] == "normal fundus")]['Right-Fundus'].sample(350,random_state=42).values

    left_diab = data.loc[(data.C == 0) & (data.LD == 1)]["Left-Fundus"].values
    right_diab = data.loc[(data.C == 0) & (data.RD == 1)]["Right-Fundus"].values

    left_glaucoma = data.loc[(data.C == 0) & (data.LG == 1)]["Left-Fundus"].values
    right_glaucoma = data.loc[(data.C == 0) & (data.RG == 1)]["Right-Fundus"].values

    left_hyper = data.loc[(data.C == 0) & (data.LH == 1)]["Left-Fundus"].values
    right_hyper = data.loc[(data.C == 0) & (data.RH == 1)]["Right-Fundus"].values

    left_myopia = data.loc[(data.C == 0) & (data.LM == 1)]["Left-Fundus"].values
    right_myopia = data.loc[(data.C == 0) & (data.RM == 1)]["Right-Fundus"].values

    left_age = data.loc[(data.C == 0) & (data.LA == 1)]["Left-Fundus"].values
    right_age = data.loc[(data.C == 0) & (data.RA == 1)]["Right-Fundus"].values

    left_other = data.loc[(data.C == 0) & (data.LO == 1)]["Left-Fundus"].values
    right_other = data.loc[(data.C == 0) & (data.RO == 1)]["Right-Fundus"].values

    normalones = np.concatenate((left_normal,right_normal),axis = 0);
    cataractones = np.concatenate((left_cataract_images,right_cataract_images),axis = 0);
    diabones = np.concatenate((left_diab,right_diab),axis = 0);
    glaucoma = np.concatenate((left_glaucoma,right_glaucoma),axis = 0);
    hyper = np.concatenate((left_hyper,right_hyper),axis = 0);
    myopia = np.concatenate((left_myopia,right_myopia),axis = 0);
    age = np.concatenate((left_age,right_age),axis=0);
    other = np.concatenate((left_other,right_other),axis = 0);

    return normalones,cataractones,diabones,glaucoma,hyper,myopia,age,other;

def has_condn(term, text):
    if term in text:
        return 1
    else:
        return 0

normal , cataract , diab, glaucoma , hyper , myopia , age, other = process_dataset(dataset);

print("Dataset stats::")
print("Normal ::" , len(normal))
print("Cataract ::" , len(cataract))
print("Diabetes ::" , len(diab))
print("Glaucoma ::" , len(glaucoma))
print("Hypertension ::" , len(hyper))
print("Myopia ::" , len(myopia))
print("Age Issues ::" , len(age))
print("Other ::" , len(other))

from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for column in dataset.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(dataset[column])
    label_encoders[column] = le

print(dataset.corr())

plt.figure(figsize = (15,8))
sns.heatmap(dataset.corr(), annot=True, linewidth=2, linecolor = 'lightgray')
plt.show()

from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tqdm import tqdm
import cv2
import random

dataset_dir = "/content/gdrive/MyDrive/preprocessed_images_1/"
image_size=224
labels = []
dataset = []
def data_gen(imagecategory , label):
    for img in tqdm(imagecategory):
        imgpath = os.path.join(dataset_dir,img);

        try:
            image = cv2.imread(imgpath,cv2.IMREAD_COLOR)
            image = cv2.resize(image,(image_size,image_size))
        except:
            continue;
        dataset.append([np.array(image),np.array(label)]);
    random.shuffle(dataset);

    return dataset;

dataset = data_gen(normal,0)
dataset = data_gen(cataract,1)
dataset = data_gen(diab,2)
dataset = data_gen(glaucoma,3)
dataset = data_gen(hyper,4)
dataset = data_gen(myopia,5)
dataset = data_gen(age,6)
dataset = data_gen(other,7)

len(dataset)

plt.figure(figsize=(12,7))
if len(dataset) > 0:  # Check if dataset is not empty
    for i in range(min(10, len(dataset))):  # Iterate up to 10 or dataset length
        sample = random.choice(range(len(dataset)))
        image = dataset[sample][0]
        category = dataset[sample][1]

        if category== 0:
            label = "Normal"
        elif category == 1 :
            label = "Cataract"
        elif category == 2:
            label = "Diabetes"
        elif category == 3:
            label = "Glaucoma"
        elif category == 4:
            label = "Hypertension"
        elif category == 5:
            label = "Myopia"
        elif category == 6:
            label = "Age Issues"
        else:
            label = "Other"

        plt.subplot(2,6,i+1)
        plt.imshow(image)
        plt.xlabel(label)
    plt.tight_layout()
else:
    print("Dataset is empty. Please load data first.")

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

train_x = np.array([i[0] for i in dataset]).reshape(-1,image_size,image_size,3);
train_y = np.array([i[1] for i in dataset])

x_train, x_temp, y_train, y_temp = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)


y_train_cat = to_categorical(y_train, num_classes=8)
y_val_cat = to_categorical(y_val, num_classes=8)
y_test_cat = to_categorical(y_test, num_classes=8)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

idg = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

idg_test = ImageDataGenerator(rescale=1./255)

idg.fit(x_train)
idg.fit(x_val)
idg_test.fit(x_test)

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import layers, models

image_size = 224
resnet50 = ResNet50(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))

for layer in resnet50.layers:
    layer.trainable = False

resnet50.summary()

n_inputs = resnet50.output_shape[1]

flat = layers.Flatten()(resnet50.output)
dense1 = layers.Dense(2048, activation="relu")(flat)
dropout1 = layers.Dropout(0.4)(dense1)
dense2 = layers.Dense(2048, activation="relu")(dropout1)
dropout2 = layers.Dropout(0.4)(dense2)
output = layers.Dense(8, activation="softmax")(dropout2)

final_resnet50 = tf.keras.models.Model(inputs=[resnet50.input], outputs=[output])

final_resnet50.summary()

count = 0
for layer in final_resnet50.layers:
  count = count +1
print(count)

from tensorflow.keras.optimizers import SGD

from sklearn.utils.class_weight import compute_class_weight
class_counts = np.bincount(y_train)
total_samples = np.sum(class_counts)
class_weights = total_samples / (len(class_counts) * class_counts)
class_weight = dict(enumerate(class_weights))

final_resnet50.compile(optimizer=SGD(learning_rate=3e-4, momentum=0.9),
              loss="categorical_crossentropy",
              metrics=['accuracy'])

train_data = (x_train, y_train_cat)
validation_data = (x_val, y_val_cat)
test_data = (x_test, y_test_cat)

history = final_resnet50.fit(train_data[0], train_data[1], validation_data=(validation_data[0], validation_data[1]), batch_size=32, epochs=50, class_weight=class_weight)

train_loss, train_accuracy = final_resnet50.evaluate(train_data[0], train_data[1], verbose=0)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

val_loss, val_accuracy = final_resnet50.evaluate(validation_data[0], validation_data[1], verbose=0)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

test_loss, test_accuracy = final_resnet50.evaluate(test_data[0], test_data[1], verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred = []
for i in final_resnet50.predict(x_test):
    y_pred.append(np.argmax(np.array(i)).astype("int32"))

print(y_pred)

for i in range(20):
    sample = random.choice(range(len(x_test)))
    image = x_test[sample]
    category = y_test[sample]
    pred_category = y_pred[sample]

    if category== 0:
        label = "Normal"
    elif category == 1 :
        label = "Cataract"
    elif category == 2:
        label = "Diabetes"
    elif category == 3:
        label = "Glaucoma"
    elif category == 4:
        label = "Hypertension"
    elif category == 5:
        label = "Myopia"
    elif category == 6:
        label = "Age Issues"
    else:
        label = "Other"

    if pred_category== 0:
        pred_label = "Normal"
    elif pred_category == 1 :
        pred_label = "Cataract"
    elif pred_category == 2:
        pred_label = "Diabetes"
    elif pred_category == 3:
        pred_label = "Glaucoma"
    elif pred_category == 4:
        pred_label = "Hypertension"
    elif pred_category == 5:
        pred_label = "Myopia"
    elif pred_category == 6:
        pred_label = "Age Issues"
    else:
        pred_label = "Other"

    plt.subplot(4,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout()

final_resnet50.save("ODIR_resnet50.h5")

final_resnet50.save('ODIR_resnet50.keras')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

class_names = ['Normal', 'Cataract', 'Diabetes', 'Glaucoma', 'Hypertension', 'Myopia', 'Age Issues', 'Other']
y_pred = final_resnet50.predict(x_test)

y_pred_classes = np.argmax(y_pred, axis=1)

y_true_classes = np.argmax(y_test_cat, axis=1)

report = classification_report(y_true_classes, y_pred_classes)
print(report)

!pip install lime

from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

explainer = lime_image.LimeImageExplainer()

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

def preprocess_image(img, target_size=(224, 224)):
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize if your model expects it
    return img

# Set the path to your dataset
dataset_path = '/content/gdrive/MyDrive/preprocessed_images_1/'

# Example: Set the path to an image for testing
image_path = dataset_path + '0_right.jpg'  # Adjust to match your file structure

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

import numpy as np
from PIL import Image

# ... (your existing code) ...

# Set the path to your dataset - Correcting the path to Google Drive
dataset_path = '/content/drive/MyDrive/preprocessed_images_1/'

# Example: Set the path to an image for testing
image_path = dataset_path + '0_right.jpg'  # Adjust to match your file structure

# Load the image
test_image_array = np.array(Image.open(image_path))

# Preprocess the test image
input_img = preprocess_image(test_image_array)

# ... (rest of your existing code) ...

# Preprocess the test image
input_img = preprocess_image(test_image_array)


from tensorflow import keras
model = keras.models.load_model('/content/ODIR_resnet50.keras')
from PIL import Image

def preprocess_image(img, target_size=(224, 224)):
    # Resize the image using PIL
    img = Image.fromarray(img)
    img = img.resize(target_size)
    img = np.array(img)

    # Add batch dimension and normalize
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

pred = model.predict(input_img)
pred_class = np.argmax(pred)


# Preprocess the test image
input_img = preprocess_image(test_image_array)


from tensorflow import keras
model = keras.models.load_model('/content/ODIR_resnet50.keras')
from PIL import Image
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

def preprocess_image(img, target_size=(224, 224)):
    # Resize the image using PIL
    img = Image.fromarray(img)
    img = img.resize(target_size)
    img = np.array(img)

    # Add batch dimension and normalize
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

pred = model.predict(input_img)
pred_class = np.argmax(pred)

# Explain using LIME
# Ensure the image passed to explain_instance is the preprocessed one
# and has the expected shape for the model.
explanation = explainer.explain_instance(
    input_img[0].astype('double'),  # Use input_img[0] instead of test_image_array
    model.predict,
    top_labels=1,
    hide_color=0,
    num_samples=1000
)

# Visualize the explanation
from skimage.color import label2rgb
temp, mask = explanation.get_image_and_mask(
    label=pred_class,
    positive_only=True,
    hide_rest=False,
    num_features=5,
    min_weight=0.0
)

# Display the original image with the mask overlaid
plt.imshow(mark_boundaries(input_img[0], mask)) # Use input_img[0] for display
plt.title(f'Explanation for Class {pred_class}')
plt.axis('off')
plt.show()

import shap
import numpy as np

# Assuming 'input_img' is your preprocessed image data
# and you want to use a subset of it as background data
background_data = input_img[:10]  # Select the first 10 images as background

# Initialize the SHAP explainer
explainer = shap.GradientExplainer(model, background_data)

# Explain a prediction
shap_values = explainer.shap_values(input_img)

# Visualize the explanation
shap.image_plot(shap_values, input_img)