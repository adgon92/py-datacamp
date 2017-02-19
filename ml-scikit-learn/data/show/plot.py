import matplotlib.pyplot as plt
from data.load.sklearn.dataset import get_digits

fig = plt.figure(figsize=(6, 6))

fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

digits = get_digits()

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]))

plt.show()


