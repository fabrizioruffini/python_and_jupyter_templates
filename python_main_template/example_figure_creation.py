
#  Useful info on: https://python4mpia.github.io/plotting/advanced.html



# To allow more customization, we can use an object-based way to make the plots
from matplotlib import pyplot as plt
import numpy as np

fig1 = plt.figure("Figure1 title")  # create a figure object. Figure size handling: fig = plt.figure(figsize=(6,8))
ax1 = fig1.add_subplot(1, 1, 1)  # create an axes object in the figure
ax1.plot([1, 2, 3, 4])
ax1.set_ylabel('some numbers')
# Set the font size via a keyword argument
ax1.set_title("My plot", fontsize='large')
ax1.legend()

if 1:
    # we can also do twin axes
    ax2 = fig1.add_subplot(2, 1, 1)
    ax2_twin = ax2.twinx()
    t = np.linspace(0., 10., 100)
    ax2.plot(t, t ** 2, 'b-')
    ax2.plot(t, 1000 / (t + 1), 'r-')
    ax2.set_ylabel('Density (cgs)', color='red')
    ax2_twin.set_ylabel('Temperature (K)', color='blue')
    ax2.set_xlabel('Time (s)')
plt.show()
# Pyplot keeps an internal reference to all figures unless specifically instructed to close a figure. Therefore,
# when making many plots, users may run out of memory.
# The solution is to explicitly close figures when they are no longer used:
plt.close(fig1)


# Note that is also allows us to easily make inset plots (for zooming):
fig2 = plt.figure("Figure2 title")
ax1 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = fig2.add_axes([0.72, 0.72, 0.16, 0.16])
fig2.savefig('./fig/a_new_plot.png')

plt.show()
plt.close(fig2)  #memory freeding




# ------ when using seaborn or dataframe.plot, you can pass the axis to the functions

from datetime import datetime
import pandas as pd

df = pd.DataFrame([{"Timestamp":datetime(2017,2,2, 9), "age": 30}, {"Timestamp":datetime(2017,2,2, 10), "age": 20}])
df.set_index("Timestamp", inplace=True),
print(df)
#    df.plot(marker='o', grid =True, table =True)
fig_df = plt.figure("Figure_df title")  # create a figure object
axFR = fig_df.add_subplot(1, 1, 1)  # create an axes object in the figure
df.plot(marker='o', grid =True, legend=False, title ='Plant-test', ax=axFR)
axFR.set_ylabel("Metetering [MW]")
#axFR.legend()  #adding legend
fig_df.savefig('./fig/the_df_plot.png') #, facecolor='0.95')

plt.show()
plt.close(fig_df)


# SEABORN PLOTS:
# It depends a bit on which seaborn function you are using.

# The plotting functions in seaborn are broadly divided into two classes

# - "Axes-level" functions, including regplot, boxplot, kdeplot, and many others
# - "Figure-level" functions, including lmplot, factorplot, jointplot and one or two others
# The first group is identified by taking an explicit ax argument and returning an Axes object.
# As this suggests, you can use them in an "object oriented" style by passing your Axes to them:

# Axes-level functions will only draw onto an Axes and won't otherwise mess with the figure,
# so they can coexist perfectly happily in an object-oriented matplotlib script.

#The second group of functions (Figure-level) are distinguished by the fact that the resulting
# plot can potentially include several Axes which are always organized in a "meaningful" way.
# That means that the functions need to have total control over the figure, so it isn't possible
# to plot, say, an lmplot onto one that already exists. Calling the function always initializes a
# figure and sets it up for the specific plot it's drawing.
import seaborn as sns
sns.set(style="ticks")

# Load the iris example dataset
import seaborn as sns; sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")

fig_sns, (ax1_sns, ax2_sns) = plt.subplots(2)
sns.regplot(x="sepal_width", y="sepal_length", data = iris, ax=ax1_sns)
setosa = iris.loc[iris.species == "setosa"]
virginica = iris.loc[iris.species == "virginica"]
sns.kdeplot(setosa.sepal_width, setosa.sepal_length, cmap="Reds", shade=True, shade_lowest=False, ax=ax2_sns)
plt.show()
plt.close(fig_sns)