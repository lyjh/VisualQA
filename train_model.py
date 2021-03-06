import VQA
from keras.callbacks import ModelCheckpoint, EarlyStopping

batch_size = 128

vqa = VQA.VQA()
vqa.use_word_embedding = False
vqa_model = vqa.create_model()

# image_caption_model.load_weights('weights/weights-improvement-13-2.99.hdf5')
file_name = 'weights/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(file_name, monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='acc', min_delta=0, patience=5, verbose=1, mode='auto')

train_generator = vqa.data_generator(batch_size=batch_size)
vqa_model.fit_generator(train_generator, steps_per_epoch=vqa.total_samples//batch_size, epochs=30, verbose=1, callbacks=[checkpoint, early])