"""
CNN feature extraction, PCA and k-NN on MNIST
Authored by: Omar Abdeen, Harris Aman, Abdul Hafeez Mohammad
Date: 4/1/2024
"""
import keras.datasets
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def genMatrix(true, predicted, title):
    # Generate confusion matrix
    cm = confusion_matrix(true, predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def pcaPlot(feat, true):
    # Plot the features
    plt.figure(figsize=(8, 6))
    for i in range(10):
        subset = feat[true == i]
        plt.scatter(subset[:, 0], subset[:, 1], label=i)
    plt.legend()
    plt.title('PCA of MNIST Test Set')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()


# *************************************************************
# // LOAD IN THE DATASET FROM MNIST

'''
While we can download the datasets as csv files, we can also make an API call
to load the data remotely. This is helpful incase the dataset is ever updated
Refernce TensorFlow documentation:
https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data 
'''
# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

'''
CNN expects data to be of 3-dimensions (width, height, depth) so we must reshape the data as it is is
currently in a 28x28 shape. After reshaping we receive a dataset of the shape (28,28,1).
Even though there's only one channel we still need to explicitly add that dimension
The data type should be float32 before dividing by 255 because we get decimal values after normalizing.
The original pixel values range from 0 to 255. Divide them by 255 to get them into the range of 0.0-1.0
'''
# Preprocess the images by reshaping and normalizing
train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255

'''
The values in the label column for MNIST are represented as sparse scalars which not suitable for 
multiclass classification models becuase they output probabilities per class. Because of this
we perform one-hot encoding to convert the labels into a categorical format which will work appropriately
with a loss function when compiling the model

Refernce TensorFlow documentation:
https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
Outputs A binary matrix representation of the input as a NumPy array. The class axis is placed last.
'''
# Convert labels to one-hot encoding
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)
# *************************************************************

# *************************************************************
# // DEFINE THE CNN ARCHITECTURE

'''
Referance Keras Documentation:
https://keras.io/guides/functional_api/

Following the API guide we can build a flexible CNN which can fit
the parameters outlined by the assignment documentation
'''
# 1. Input Layer
input_tensor = Input(shape=(28, 28, 1))
# 2. First Convolutional Layer
x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_tensor)
# 3. First Pooling Layer
x = MaxPooling2D((2, 2), strides=2)(x)
# 4. Second Convolutional Layer
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# 5. Second Pooling Layer
x = MaxPooling2D((2, 2), strides=2)(x)
# 6. Flatten Layer
x = Flatten()(x)
# 7. Output Layer
output_tensor = Dense(10, activation='softmax')(x)

# Create the model
model = Model(inputs=input_tensor, outputs=output_tensor)
# *************************************************************

# *************************************************************
# // COMPILE, TRAIN, EVALUATE

'''
Refernce Keras Documentation to compile the model:
https://keras.io/api/models/model_training_apis/

We are using categorical_crossentropy as the loss function. We could also
make use of SparseCategoricalCrossentropy however we coverted the labels into
a one-hot representation. While we could have left it as it is, for this specific
model and its application it is more appropriate to convert the label format

Reference TensorFlow documentation:
https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy

NOTE: The SciKitLearn library does not support a sparse input which would be a problem 
when it is time for the PCA
'''
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

'''
We've arbitrarily chosen 10 epochs and a batch size of 128. 
10 Epoch's is more or less enough for quick feedback on the model 
Batch size of 128 gives a healthy medium between memory constrains and faster learning
NOTE: Assignment outline did not request to provide a validation split, the entire training
data is being used to fit the model, however, this can be adjusted as needed.
'''
# Train the model on the training dataset
model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.0)

'''
Printing the model summary provides key details about the number of parameters by layer.

Reference Documentation:
https://keras.io/api/models/model/#summary-method
'''
# Display Model Layer Information
print(model.summary())

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'CNN Accuracy on Test Set: {test_accuracy}')
# *************************************************************

# *************************************************************
# PREDICTIONS AND GENERATE A CONFUSION MATRIX
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Generate Confusion Matrix
genMatrix(true_classes, predicted_classes, "CNN Confusion Matrix")
# *************************************************************

# *************************************************************
# PCA AND FEATURE PLOTTING

'''
To extract the feature vectors from the Flatten layer the Keras API provides
a get_layer() method which can be used to return a layer instance based on
either the layer name or it's index. Retrieve the outputs(feature vector) using
.output

Reference Keras API:
https://keras.io/api/models/model/#summary-method
'''
# Extract features from the Flatten layer
feature_model = Model(inputs=model.input, outputs=model.get_layer('flatten').output)

# Get features from model prediction on test images
features = feature_model.predict(test_images)

'''
Use SciKitLearn library for Principal Component Analysis
The PCA class allows to specify the number of components using the
n_components parameter

Reference Documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
'''
# Apply PCA to reduce to 2 components for visualization
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

# Plot the PCA with a distinct color for each class
pcaPlot(features_pca, true_classes)
# *************************************************************

# *************************************************************
# KNN Classification with 10 Principal Components

# Train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# First, extract features for both training and test images
train_features = feature_model.predict(train_images)
test_features = feature_model.predict(test_images)

# Apply PCA with 10 components to both sets of features
pca_10 = PCA(n_components=10)
train_features_pca_10 = pca_10.fit_transform(train_features) # Fit and transform on training features
test_features_pca_10 = pca_10.transform(test_features) # Only transform test features, don't fit

# Fit the KNN classifier on the PCA-transformed training features
knn.fit(train_features_pca_10, np.argmax(train_labels, axis=1)) # Use actual training labels for fitting

# Predict on the PCA-transformed test dataset
predictions_knn = knn.predict(test_features_pca_10) # Predict using PCA-transformed test features

# Calculate and display the accuracy based on the test labels
accuracy_knn = np.mean(predictions_knn == np.argmax(test_labels, axis=1))
print(f'KNN Test Accuracy with 10 PCA Components: {accuracy_knn}')

# Generate and display the confusion matrix based on the true test labels and KNN predictions
genMatrix(np.argmax(test_labels, axis=1), predictions_knn, "KNN Confusion Matrix")
