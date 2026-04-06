#!/usr/bin/env python3
"""Plot a stacked bar graph of fruit per person"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plot a stacked bar graph of the number of fruit per person"""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(3)
    plt.bar(x, fruit[0], width=0.5, color='red', label='apples')
    plt.bar(x, fruit[1], width=0.5, color='yellow', label='bananas',
            bottom=fruit[0])
    plt.bar(x, fruit[2], width=0.5, color='#ff8000', label='oranges',
            bottom=fruit[0] + fruit[1])
    plt.bar(x, fruit[3], width=0.5, color='#ffe5b4', label='peaches',
            bottom=fruit[0] + fruit[1] + fruit[2])
    plt.xticks(x, ['Farrah', 'Fred', 'Felicia'])
    plt.yticks(np.arange(0, 81, 10))
    plt.ylim(0, 80)
    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.legend()
    plt.show()
