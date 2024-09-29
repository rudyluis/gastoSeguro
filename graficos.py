import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Some example data to display
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

# Crear datos para gráficos de cajas
data_boxplot = [y, y, -y, -y]

fig, axs = plt.subplots(2, 2)
sns.boxplot(data=data_boxplot, ax=axs[0, 0])
axs[0, 0].set_title('Boxplot [0, 0]')

sns.boxplot(data=data_boxplot, ax=axs[0, 1], orient='h')
axs[0, 1].set_title('Boxplot [0, 1]')

sns.boxplot(data=data_boxplot, ax=axs[1, 0])
axs[1, 0].set_title('Boxplot [1, 0]')

sns.boxplot(data=data_boxplot, ax=axs[1, 1], orient='h')
axs[1, 1].set_title('Boxplot [1, 1]')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')



plt.tight_layout()
plt.show()
