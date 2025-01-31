import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from IPython.display import display, clear_output

def select_area(matrix):
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='viridis')
    fig.colorbar(cax)

    selected_matrix = None

    # Function to handle the selection
    def onselect(eclick, erelease):
        nonlocal selected_matrix
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        selected_matrix = matrix[y1:y2, x1:x2]
        print("Selected Matrix:\n", selected_matrix)

    # Function to handle the button click
    def accept(event):
        plt.close(fig)
        clear_output(wait=True)
        print("Final Selected Matrix:\n", selected_matrix)

    # Connect the rectangle selector
    rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                      interactive=True)

    # Add a button to accept the selection
    ax_button = plt.axes([0.81, 0.01, 0.1, 0.075])
    button = Button(ax_button, 'Accept')
    button.on_clicked(accept)

    plt.show()

    return selected_matrix