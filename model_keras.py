from keras.callbacks import EarlyStopping
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout
from keras.models import Model
import numpy as np

import load_data_cv

np.random.seed(100000)
# i = int(121 * 0.8)
# print(i)
# i = 100
X_train_text, X_test_text, X_train_audio, X_test_audio, X_train_gest, X_test_gest, X_train_video, X_test_video, Y_train, Y_test = load_data_cv.load(
    1)

print(X_train_audio.shape, X_test_audio.shape)

batch_size = 50
text_input_dim = 300
audio_input_dim = 300
video_input_dim = 300
gestures_input_dim = 39
classes = 2
hidden_size = (1024, 512)

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
video_input = Input(shape=(video_input_dim,), name='video_input')
audio_input = Input(shape=(audio_input_dim,), name='audio_input')
text_input = Input(shape=(text_input_dim,), name='text_input')
me_input = Input(shape=(gestures_input_dim,), name='me_input')

x = Concatenate(name='concat_input')([audio_input, video_input, text_input, me_input])
x = Dropout(0.5)(x)

h1 = Dense(hidden_size[0], activation='relu')(x)
h1 = Dropout(0.5)(h1)

y = Dense(2, activation='sigmoid')(h1)

model = Model(inputs=[video_input, audio_input, text_input, me_input], outputs=y)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
calls = [
    # EarlyStopping(monitor='val_loss', patience=20)
]
# Train the model
model.fit([X_train_video, X_train_audio, X_train_text, X_train_gest], Y_train, batch_size=batch_size, epochs=2000,
          validation_data=([X_test_video, X_test_audio, X_test_text, X_test_gest], Y_test), verbose=2, callbacks=calls)
model.save('model_all.h5')
