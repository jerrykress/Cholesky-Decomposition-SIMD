from numpy import promote_types
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored

filename = "test_result.png"

data = pd.read_csv("test_results.txt", sep=" ", header=None)
data.columns = ["Input Size", "Sum Error", "Percentage", "Seq Time", "Neon Time"]
data["Speed Up"] = data["Seq Time"] / data["Neon Time"]
data["Increment"] = (
    data["Speed Up"] - data.shift(periods=1, axis="index", fill_value=0)["Speed Up"]
)

fig, axes = plt.subplots(nrows=2, ncols=2)
plt.tight_layout(pad=4, w_pad=4.0, h_pad=5)

################ plot 1 ####################
axes[0, 0].title.set_text("Error Percentage")
axes[0, 0].set_ylabel("%")
data.plot(ax=axes[0, 0], x="Input Size", y="Percentage", kind="line")

################ plot 2 ####################
axes[0, 1].title.set_text("Execution Time")
axes[0, 1].set_ylabel("ms")
data.plot(ax=axes[0, 1], x="Input Size", y=["Seq Time", "Neon Time"], kind="line")

################ plot 3 ####################
axes[1, 0].title.set_text("Speed Up Factor")
axes[1, 0].set_ylabel("times")
data.plot(ax=axes[1, 0], x="Input Size", y="Speed Up", kind="line")

################ plot 4 ####################
axes[1, 1].title.set_text("Speed Up Increment")
axes[1, 1].set_ylabel("times")
data.plot(ax=axes[1, 1], x="Input Size", y="Increment", kind="line")

plt.savefig(filename)

msg = "[TEST] Graph saved to: " + filename
print(colored(msg, "blue"))

avg_percent = data["Percentage"].mean()
print("Average error percentage: ", avg_percent)
