import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('mixture.csv')
# print(data)
print(data['Y'])        


# DaFr = data.plot.scatter(x='X1', y='X2', c=data['Y'])

plt.scatter(data.X1, data.X2, c=data.Y)


plt.show()

