import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.figure(figsize=(6,6))

def displot(n, title):
    sns.distplot(n)
    plt.title(title)
    plt.show()

def countplot(x, data, title):
    plt.figure(figsize=(6,6))
    sns.countplot(x=x, data=data)
    plt.title(title)
    plt.show()


def heatmap(data, title):
    corr = data.corr()
    # Generate a heatmap using seaborn
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(title)
    plt.show()