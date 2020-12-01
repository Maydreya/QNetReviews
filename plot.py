from pickle import load
import matplotlib.pyplot as plt

with open('trainHistoryold3', 'rb') as handle:
    oldhistory = load(handle)


plt.plot(oldhistory['loss'],
            label='Доля потерь на обучающем наборе')
plt.plot(oldhistory['val_loss'],
            label='Доля потерь на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля потерь')
plt.legend()
plt.show()



