from PIL import Image
import numpy as np
import pickle
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten, Reshape, Dropout
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os.path

EMBEDDING_DIM = 300

class VQA():

    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index2word = None
        self.word2index = None
        self.answer2index = pickle.load(open('answer2index.p', 'rb'))
        self.index2answer = pickle.load(open('index2answer.p', 'rb'))
        self.answer_size = len(self.answer2index)
        self.total_samples = None
        self.use_word_embedding = False
        self.encoded_images = pickle.load( open( "encoded_images_COCO_inceptionV3.p", "rb" ) )
        self.variable_initializer()

    def variable_initializer(self):
        df = pd.read_csv('vqa_dataset_v2.txt', delimiter='\t')
        df = df.dropna()
        self.total_samples = df.shape[0]
        # iter = df.iterrows()
        # max_len = 0
        # for i in range(self.total_samples):
        #     x = next(iter)df = 
        #     q_len = len(x[1][1].split())
        #     if q_len > max_len:
        #         max_len = q_len
        # self.max_cap_len = max_len
        self.max_cap_len = 23
        print ("Total samples : "+str(self.total_samples))

        unique = pickle.load(open('unique.p', 'rb'))
        self.vocab_size = len(unique)
        self.word2index = {}
        self.index2word = {}
        for i, word in enumerate(unique):
            self.word2index[word]=i
            self.index2word[i]=word
        print ("Vocabulary size: "+str(self.vocab_size))
        print ("Maximum caption length: "+str(self.max_cap_len))
        print ("Variables initialization done!")

    def data_generator(self, batch_size = 32):
        encoded_questions = []
        answers = []
        images = []
        
        df = pd.read_csv('vqa_dataset_v2.txt', delimiter='\t')
        df = df.dropna()
        df = df.sample(frac=1)
        questions = np.array(df['question'])
        imgs = np.array(df['image_id'])
        labels = np.array(df['answer'])

        count = 0
        while True:
            for i in range(len(labels)):
                ans = labels[i]
                question = questions[i]
                img = imgs[i]
                current_image = self.encoded_images[img]
                count+=1
                
                encoded_q = self.encode_question(question)
                encoded_questions.append(encoded_q)
                
                a = np.zeros(self.answer_size)
                a[ans] = 1
                answers.append(a)
                
                images.append(current_image)

                if count>=batch_size:
                    answers = np.asarray(answers)
                    images = np.asarray(images)
                    encoded_questions = sequence.pad_sequences(encoded_questions, maxlen=self.max_cap_len, padding='post')
                    yield [[images, encoded_questions], answers]
                    encoded_questions = []
                    answers = []
                    images = []
                    count = 0

    def get_or_load_embedding_matrix(self, picklefile, glovefile):
        if not os.path.exists(picklefile):
            embeddings_index = {}
            f = open(glovefile, encoding='utf8')
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

            embedding_matrix = np.zeros((self.vocab_size, EMBEDDING_DIM))
            for word, i in self.word2index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            with open(picklefile, "wb") as encoded_pickle:
                pickle.dump(embedding_matrix, encoded_pickle)
        embedding_matrix = pickle.load(open(picklefile,'rb'))
        return embedding_matrix

    def create_model(self):
        image_model = Sequential()
        image_model.add(Reshape((2048,), input_shape=(2048,)))

        if self.use_word_embedding:
            embedding_matrix = self.get_or_load_embedding_matrix('glove_embedding_matrix.p', 'glove.42B.300d.txt')
        
            language_model = Sequential([
                    Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.max_cap_len, weights=[embedding_matrix], trainable=False),
                    LSTM(256, return_sequences=True),
                    LSTM(256, return_sequences=True),
                    LSTM(256, return_sequences=False)
                ])

        else:
            language_model = Sequential([
                    Embedding(self.vocab_size, EMBEDDING_DIM, input_length=self.max_cap_len, trainable=True),
                    LSTM(256, return_sequences=True),
                    LSTM(256, return_sequences=True),
                    LSTM(256, return_sequences=False)
                ])
        
        final_model = Sequential()
        final_model.add(Merge([image_model, language_model], mode='concat', concat_axis=1))

        for _ in range(3):
            final_model.add(Dense(512, kernel_initializer='uniform'))
            final_model.add(Activation('tanh'))
            final_model.add(Dropout(0.5))
        
        final_model.add(Dense(self.answer_size))
        final_model.add(Activation('softmax'))

        final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        final_model.summary()
        return final_model

    def encode_question(self, question):
        return [(self.word2index[txt] if txt in self.word2index else self.word2index['<Unk>']) for txt in question[:-1].split()]

    def decode_answer(self, ans):
        return self.index2answer[ans]