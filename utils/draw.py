import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))



# voxels perspective
# Np = [x, y, z]
# X: 4, 1, 1: left / right
# Y: 1, 4, 1: back / front
# Z: 1, 1, 4: up / down


def draw_phenotype(phenotype, id_individual, CUBE_FACE_SIZE, ranking, fitness, path, voxel_types, voxel_types_colors):

    # Define color map for values in body
    color_map = {
        voxel_id: tuple(c / 255 for c in voxel_types_colors[name]) + (0.5,)
        for name, voxel_id in voxel_types.items()
    }

    # Function to draw a single 1x1x1 cube
    def draw_cube(ax, position, color):
        x, y, z = position
        vertices = np.array([
            [x, y, z],
            [x + 1, y, z],
            [x + 1, y + 1, z],
            [x, y + 1, z],
            [x, y, z + 1],
            [x + 1, y, z + 1],
            [x + 1, y + 1, z + 1],
            [x, y + 1, z + 1]
        ])
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
        ]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='k', linewidths=0.5))

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Iterate through body and draw cubes
    x_dim, y_dim, z_dim = phenotype.shape
    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                val = phenotype[x, y, z]
                if val > 0 and val in color_map:
                    draw_cube(ax, (y, x, z), color_map[val]) # invert y and x to match voxcraft-viz

    # Set limits to match array shape exactly
    ax.set_xticks(np.arange(0, CUBE_FACE_SIZE + 1, 1))
    ax.invert_yaxis()  # to match voxcraft-viz
    ax.set_yticks(np.arange(0, CUBE_FACE_SIZE+ 1, 1))
    ax.set_zticks(np.arange(0, CUBE_FACE_SIZE+ 1, 1))

    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    #ax.set_title('Voxel Visualization (1x1x1 Cubes, Correct Colors)')
    plt.tight_layout()

    # Save the image
    plt.savefig(f"{path}/{ranking}_{fitness}_{id_individual}.png", dpi=300)
    plt.close(fig)

