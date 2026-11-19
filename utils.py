import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from os.path import join as pjoin


def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to set for random number generation.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # every GPU; does nothing on a CPU-only machine


# Define the kinematic tree for connecting joints
kinematic_tree = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20]
]

def plot_3d_motion(save_path, joints, title, figsize=(10, 10), fps=120, radius=4):
    # Split the title if it's too long
    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # Reshape the joints data
    data = joints.copy().reshape(len(joints), -1, 3)
    # fig = plt.figure(figsize=figsize)
    # ax = p3.Axes3D(fig)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    init()

    # Compute min and max values for the data
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    # Define colors for the kinematic tree
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    frame_number = data.shape[0]

    # Adjust the height offset
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    # Center the data
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        # Clear existing lines and collections
        for line in ax.lines:
            line.remove()
        for collection in ax.collections:
            collection.remove()

        # Update the view
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5

        # Plot the XZ plane
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

        # Plot the trajectory
        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1] - trajec[index, 1], linewidth=1.0, color='blue')

        # Plot the kinematic tree
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth, color=color)
        # Hide axis labels
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Create the animation
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # Save the animation
    ani.save(save_path, fps=fps)
    plt.close()

    print(f'Animation saved to {save_path}!')


def plot_metrics(val_metrics, save_path, show_plot=True):
    """
    Plot the validation metrics over epochs and save the plot.

    Parameters
    ----------
    val_metrics : list of dict
        A list containing dictionaries of validation metrics for each epoch.
    save_path : str
        The path where the plot will be saved.
    """
    list_bleu = [score['BLEU'] for score in val_metrics]
    list_cider = [score['CIDEr'] for score in val_metrics]
    list_meteor = [score['METEOR'] for score in val_metrics]
    list_rouge1 = [score['ROUGE-1'] for score in val_metrics]
    list_rouge2 = [score['ROUGE-2'] for score in val_metrics]
    list_rougeL = [score['ROUGE-L'] for score in val_metrics]

    plt.plot(list_bleu, label='BLEU')
    plt.plot(list_cider, label='CIDEr')
    plt.plot(list_meteor, label='METEOR')
    plt.plot(list_rouge1, label='ROUGE-1')
    plt.plot(list_rouge2, label='ROUGE-2')
    plt.plot(list_rougeL, label='ROUGE-L')

    # Add title and labels
    plt.title('Validation Metrics Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.legend()
    
    # Save the plot to the specified path
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()