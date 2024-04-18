from PIL import Image

from keras.preprocessing import image
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
import cv2

model = load_model('model.h5')

source = cv2.VideoCapture(0)

im_shape = (128, 128)

TRAINING_DIR = './archive/train'
TEST_DIR = './archive/test'

seed = 10

BATCH_SIZE = 64

data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generator para parte train
train_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=im_shape, shuffle=True, seed=seed,
                                                    class_mode='categorical', batch_size=BATCH_SIZE, subset="training")
# Generator para parte validação
validation_generator = val_data_generator.flow_from_directory(TRAINING_DIR, target_size=im_shape, shuffle=False, seed=seed,
                                                    class_mode='categorical', batch_size=BATCH_SIZE, subset="validation")

test_generator = ImageDataGenerator(rescale=1./255)
test_generator = test_generator.flow_from_directory(TEST_DIR, target_size=im_shape, shuffle=False, seed=seed,
                                                    class_mode='categorical', batch_size=BATCH_SIZE)


nb_train_samples = train_generator.samples
nb_validation_samples = validation_generator.samples
nb_test_samples = test_generator.samples
classes = list(train_generator.class_indices.keys())
print('Classes: '+str(classes))
num_classes  = len(classes)


while (True):

    # Capture the video frame
    # by frame
    if not source.isOpened():
        break

    ret, img = source.read()

    # Display the resulting frame
    cv2.imshow('frame', img)

    img = cv2.resize(img, (im_shape[0], im_shape[1]))
        
    # Converta a imagem para um array numpy
    img_array = image.img_to_array(img)

    # Adicione uma dimensão extra para representar o batch (necessário para o modelo)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize os valores dos pixels para o intervalo [0, 1]
    img_array /= 255.0

    # Faça a predição
    prediction = model.predict(img_array)

    # Converta as probabilidades preditas em uma classe
    predicted_class = np.argmax(prediction)

    # Obtenha o nome da classe correspondente
    predicted_class_name = classes[predicted_class]

    print(f"A imagem é classificada como: {predicted_class_name}")


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
