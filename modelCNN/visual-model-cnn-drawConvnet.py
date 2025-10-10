import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle

# ----------------------
# Paramètres généraux
NumDots = 4
NumConvMax = 6
NumFcMax = 20
fc_unit_size = 5
layer_width = 120  # espacement horizontal
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Darker = 0.15
Black = 0.

# ----------------------
# Fonctions de dessin
def add_layer_with_omission(patches, colors, size=(24,24), num=5, num_max=6, num_dots=4, top_left=[0,0], loc_diff=[3,-3]):
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    this_num = min(num, num_max)
    start_omit = (this_num - num_dots)//2
    end_omit = this_num - start_omit
    start_omit -= 1
    for ind in range(this_num):
        omit = (num > num_max) and (start_omit < ind < end_omit)
        if omit:
            patches.append(Circle(loc_start + ind*loc_diff + np.array(size)/2, 0.5))
            colors.append(Black)
        else:
            patches.append(Rectangle(loc_start + ind*loc_diff, size[1], size[0]))
            colors.append(Medium if ind % 2 else Light)

def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn, top_left_list, loc_diff_list, num_show_list, size_list):
    start_loc = top_left_list[ind_bgn] + (num_show_list[ind_bgn]-1)*np.array(loc_diff_list[ind_bgn]) \
                + np.array([start_ratio[0]*(size_list[ind_bgn][1]-patch_size[1]),
                            -start_ratio[1]*(size_list[ind_bgn][0]-patch_size[0])])
    end_loc = top_left_list[ind_bgn+1] + (num_show_list[ind_bgn+1]-1)*np.array(loc_diff_list[ind_bgn+1]) \
              + np.array([end_ratio[0]*size_list[ind_bgn+1][1],
                          -end_ratio[1]*size_list[ind_bgn+1][0]])
    patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0]))
    colors.append(Dark)
    for dx in [0, patch_size[1]]:
        for dy in [0, -patch_size[0]]:
            patches.append(Line2D([start_loc[0]+dx, end_loc[0]], [start_loc[1]+dy, end_loc[1]]))
            colors.append(Darker)

def label_above(xy, text, xy_off=[0,10]):
    """Affiche le texte au-dessus du bloc"""
    plt.text(xy[0]+xy_off[0], xy[1]+xy_off[1], text, family='sans-serif', size=8, ha='center')

# ----------------------
# Script principal
if __name__ == '__main__':
    patches = []
    colors = []
    fig, ax = plt.subplots()

    # ----------------------
    # Conv blocks
    size_list = [(60,60), (52,52), (48,48), (42,42), (38,38)]
    num_list = [32, 64, 64, 128, 256]
    x_diff_list = [0, layer_width, layer_width+20, layer_width, layer_width]
    y_offsets = [0, -15, 0, -15, 0]
    top_left_list = np.c_[np.cumsum(x_diff_list), y_offsets]
    loc_diff_list = [[3,-3]]*len(size_list)
    num_show_list = list(map(min, num_list, [NumConvMax]*len(num_list)))

    conv_props = [
        "32 filtres, 3x3, ReLU",
        "64 filtres, 3x3, ReLU + Dropout0.1",
        "64 filtres, 3x3, ReLU",
        "128 filtres, 3x3, ReLU + Dropout0.2",
        "256 filtres, 3x3, ReLU + Dropout0.2"
    ]

    # ----------------------
    # Input Layer (aligné au premier bloc)
    input_size = (60, 60)
    input_top_left = top_left_list[0] - np.array([layer_width, 0])  # à gauche du premier Conv Block
    patches.append(Rectangle(input_top_left, input_size[1], input_size[0]))
    colors.append(Light)
    label_above(input_top_left + np.array([input_size[1] / 2, 0]), "Input\n150x150x1")

    # ----------------------
    # Dessiner les Conv blocks
    for ind in range(len(size_list)-1,-1,-1):
        add_layer_with_omission(patches, colors, size=size_list[ind], num=num_list[ind],
                                num_max=NumConvMax, num_dots=NumDots,
                                top_left=top_left_list[ind], loc_diff=loc_diff_list[ind])
        label_above(top_left_list[ind] + np.array([size_list[ind][1]/2,0]),
                    f"Conv Block {ind+1}\n{conv_props[ind]}")

    # ----------------------
    # Connexions
    start_ratio_list = [[0.4,0.5]]*4
    end_ratio_list = [[0.4,0.5]]*4
    patch_size_list = [(5,5)]*4
    for ind in range(4):
        add_mapping(patches, colors, start_ratio_list[ind], end_ratio_list[ind], patch_size_list[ind], ind,
                    top_left_list, loc_diff_list, num_show_list, size_list)

    # ----------------------
    # Fully connected
    fc_sizes = [6400, 128, 1]
    size_list_fc = [(fc_unit_size, fc_unit_size)]*3
    num_list_fc = fc_sizes
    num_show_list_fc = list(map(min, num_list_fc, [NumFcMax]*len(num_list_fc)))
    x_diff_list_fc = [sum(x_diff_list)+layer_width, layer_width, layer_width]
    y_offsets_fc = [0, -10, 0]
    top_left_list_fc = np.c_[np.cumsum(x_diff_list_fc), y_offsets_fc]
    loc_diff_list_fc = [[fc_unit_size, -fc_unit_size]]*len(top_left_list_fc)
    text_list_fc = ['Flatten', 'Dense', 'Output']
    fc_props = [
        "Flatten",
        "Dense 128 units, ReLU + Dropout0.2",
        "Dense 1 unit, Sigmoid"
    ]

    for ind in range(len(size_list_fc)):
        add_layer_with_omission(patches, colors, size=size_list_fc[ind], num=num_list_fc[ind],
                                num_max=NumFcMax, num_dots=NumDots,
                                top_left=top_left_list_fc[ind], loc_diff=loc_diff_list_fc[ind])
        label_above(top_left_list_fc[ind] + np.array([size_list_fc[ind][1]/2,0]),
                    f"{text_list_fc[ind]}\n{fc_props[ind]}")

    # ----------------------
    # Dessiner patches et lignes
    for patch, color in zip(patches, colors):
        patch.set_color(color*np.ones(3))
        if isinstance(patch, Line2D):
            ax.add_line(patch)
        else:
            patch.set_edgecolor(Black*np.ones(3))
            ax.add_patch(patch)

    fig.set_size_inches(18,6)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
