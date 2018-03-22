import glob
from PIL import Image
import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, RMSprop
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import VQA

unique = pickle.load(open('unique.p', 'rb'))

vqa = VQA.VQA()
max_cap_len = vqa.max_cap_len
vqa_model = vqa.create_model()

vqa_model.load_weights('weights/weights-improvement-01-3.14.hdf5')

model = InceptionV3(weights='imagenet')
new_input = model.input
hidden_layer = model.layers[-2].output

featurize_model = Model(new_input, hidden_layer)

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

def featurize_image(image):
    image = preprocess(image)
    temp_enc = featurize_model.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc

def predict_answer(image_path, question):
    img_feat = featurize_image(image_path)
    Image.open(image_path).show()
    ques_feat = vqa.encode_question(question)
    ques_feat = sequence.pad_sequences([ques_feat], maxlen=max_cap_len, padding='post')

    preds = vqa_model.predict([np.array([img_feat]), np.array(ques_feat)])[0]
    args = np.argsort(preds)[-3:][::-1]
    for i in range(len(args)):
        print ("'{0}' with probability of {1}".format(vqa.decode_answer(args[i]), preds[args[i]]))