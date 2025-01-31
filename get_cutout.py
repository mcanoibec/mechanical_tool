import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from backend.import_xyz_img import import_xyz_img
import pandas as pd

def select_area(file=None):
    if file!=None:
        topography_file=r'c:\Users\mcano\Code2\data\raw\EFM4\EFM4 Topography Flattened.txt'
        matrix, x, y = import_xyz_img(topography_file)
    else:
        matrix=np.loadtxt(r'temporary\rough_topo.txt')
    mat_idxs=np.zeros_like(matrix)
    mat_idxs=mat_idxs.reshape(len(matrix)**2)
    mat_idxs=pd.DataFrame(mat_idxs)
    mat_idxs=np.array(mat_idxs.index).reshape(len(matrix), len(matrix))

    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='afmhot')
    fig.colorbar(cax)


    selected_matrix = None
    selected_idx = None
    # Function to handle the selection
    def onselect(eclick, erelease):
        nonlocal selected_matrix
        nonlocal selected_idx
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        selected_matrix = matrix[y1:y2, x1:x2]
        selected_idx= mat_idxs[y1:y2, x1:x2]
        selected_idx=(selected_idx.reshape(len(selected_idx[0])*len(selected_idx[:,0]))).astype(int)
        print(rf'Current selection:\n x: ({x1},{x2}), y: ({y1},{y2})')

    # Function to handle the button click
    def accept(event):
        plt.close(fig)

    # Connect the rectangle selector
    rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                      button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                      interactive=True)

    # Add a button to accept the selection
    ax_button = plt.axes([0.81, 0.01, 0.1, 0.075])
    button = Button(ax_button, 'Accept')
    button.on_clicked(accept)

    plt.show()

    return selected_idx

# If using own file

selected_matrix = select_area()
np.savetxt(r'temporary\cutout_idx.txt', selected_matrix)
print(rf'Final selection:\n ({selected_matrix})')
