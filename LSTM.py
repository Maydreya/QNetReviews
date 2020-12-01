from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras import utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt
import pickle

num_words = 10000
max_review_len = 1000
classes = 3

# Читаем данные из data.csv
train = pd.read_csv('trainnorm.csv',
                    skiprows=1,
                    header=None,
                    names=['Review'])
train2 = pd.read_csv('reviews/train.csv',
                     header=None,
                     names=['Review', 'Class'])

# Выделяем данные для обучения
reviews = train['Review']

# Выделяем правильные ответы
y_train = utils.to_categorical(train2['Class'] + 1, classes)

# Представляем текст в виде набора чисел по частоте использования, 1 = самое частое слово
tokenizer = Tokenizer(num_words=num_words)

# Обучаем tokenizer на отзывах Кинопоиск
tokenizer.fit_on_texts(reviews)

# Преобразоываваем отзывы в числовую последовательность
sequences = tokenizer.texts_to_sequences(reviews)

# Ограничиваем длину отзывов
x_train = pad_sequences(sequences, maxlen=max_review_len)

# Создаем нейронную сеть
model = Sequential()

# Вектор представления слов Embedding
model.add(Embedding(num_words, 32, input_length=max_review_len))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# Обучаем нейронную сеть
model_save_path = 'best_model3.h5'
# Создаем callback для сохранения нейронной сети на каждой эпохе, если качество работы
# на проверочном наборе данных улучшилось. Сеть сохраняется в файл best_model.h5
checkpoint_callback = ModelCheckpoint(
    filepath=model_save_path,
    monitor='val_loss',
    mode='min',
    save_best_only=True
)

history = model.fit(x_train,
                    y_train,
                    epochs=100,
                    batch_size=1500,
                    validation_split=0.2,
                    callbacks=[checkpoint_callback])

print('\nhistory dict:', history.history)

with open('trainHistoryold3', 'wb') as handle:
    pickle.dump(history.history, handle)
# Загружаем модель с лучшей долей правильных ответов на проверочном наборе данных
model.load_weights(model_save_path)
plt.plot(history.history['loss'],
         label='Доля потерь на обучающем наборе')
plt.plot(history.history['val_loss'],
         label='Доля потерь на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля потерь')
plt.legend()
plt.show()

# Загружаем набор данных для тестирования
test = pd.read_csv('testnorm.csv',
                   header=None,
                   skiprows=1,
                   names=['Review'])
test2 = pd.read_csv('reviews/test.csv',
                    header=None,
                    names=['Review', 'Class'])

# Преобразуем отзывы в числовое представление
testreviews = test['Review']
# Используем токинезатор, обученный на наборе данных train, чтобы использовался один и тот же словарь
test_sequences = tokenizer.texts_to_sequences(testreviews)

# Ограничиваем длину последовательности
x_test = pad_sequences(test_sequences, maxlen=max_review_len)

# Правильные ответы
y_test = utils.to_categorical(test2['Class'] + 1, classes)

# Оцениваем качество работы сети на тестовом наборе
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)
with open('tokenizer3.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
