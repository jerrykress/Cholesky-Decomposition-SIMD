from numpy import promote_types
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored

data = pd.read_csv("test_results.txt", sep=" ", header=None)
data.columns = ["Input Size", "Sum Error", "Percentage"]

data.plot(x="Input Size", y="Percentage", kind="line")

filename = "test_result.png"
plt.savefig(filename)
msg = "[TEST] Graph saved to: " + filename
print(colored(msg, "blue"))

avg_percent = data["Percentage"].mean()

print("Average error percentage: ", avg_percent)
