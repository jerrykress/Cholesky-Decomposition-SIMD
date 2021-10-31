import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('test_results.txt', sep=" ", header=None)
data.columns = ["Input Size", "Sum Error", "Percentage"]

data.plot(x ='Input Size', y='Percentage', kind = 'line')
plt.show()