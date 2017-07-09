import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from keras.preprocessing import image
from keras.utils import to_categorical

# images.
img_width, img_height = 150, 150
train_data_dir = 'flowers/train'
validation_data_dir = 'flowers/validation'

# path to the model weights files.
top_model_weights_path = 'bottleneck_fc_model.h5'

nb_train_samples = 1000
nb_validation_samples = 800
epochs = 20
batch_size = 16

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train_category.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_train_category.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train_category.npy'))
    # assign labels according to number of images per class
    train_labels = np.array([[0]*200,[1]*200,[2]*200,[3]*200,[4]*200])
    train_labels_one_hot = to_categorical(train_labels, 5)

    validation_data = np.load(open('bottleneck_features_validation_category.npy'))
    # assign labels according to number of images per class
    validation_labels = np.array([[0]*160,[1]*160,[2]*160,[3]*160,[4]*160])
    validation_data_one_hot = to_categorical(validation_labels, 5)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels_one_hot[:len(train_data)],
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_data_one_hot[:len(validation_data)]))
    model.save_weights(top_model_weights_path)


def predict_image_class(test_image):
    base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(150,150,3))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(5, activation='sigmoid'))
    top_model.load_weights(top_model_weights_path)

    final_input = Input(shape=(150,150,3))
    x = base_model(final_input)
    result = top_model(x)
    final_model = Model(input=final_input, output=result)

    img = Image.open(test_image)
    img = img.resize((150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = final_model.predict(x)
    print preds


# save_bottlebeck_features()
# model = train_top_model()
test_image = 'flowers/test/daisy/daisy_601.jpg'
predict_image_class(test_image)