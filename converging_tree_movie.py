from levelstat import level_stat_orbit
from treeplot import treeplot
from levelstat import levelstat_plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def make_movie(necklace_base, k_start, k_end, depth, bar_max=None):
    levels_all = list()
    adj_matrix_all = list()
    necklace_all = list()
    k_all = list()
    bar_max_computed = 0
    for k in range(k_start, k_end + 1):
        necklace = necklace_base * k
        levels, adj_matrix = level_stat_orbit(necklace, depth=depth + 1)
        levels_all.append(levels)
        adj_matrix_all.append(adj_matrix)
        necklace_all.append(necklace)
        k_all.append(k)
        bar_max_computed = max(bar_max_computed, max(levels))

    if bar_max is None:
        bar_max = bar_max_computed

    # Create a figure and an axis
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Set up the writer
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)

    # Create the video
    with writer.saving(fig, f"trees_animation_{necklace_base}.mp4", 100):
        for necklace, levels, adj_matrix, k in zip(necklace_all, levels_all, adj_matrix_all, k_all):
            ax[0].clear()
            ax[1].clear()
            fig.suptitle(f"Halsband: ({necklace_base})^{k}\n{levels}")
            # ax[1].set_title(str(levels))
            treeplot(adj_matrix, ax=ax[0])
            levelstat_plot(levels, ax=ax[1], bar_max=bar_max)
            writer.grab_frame()
    

# make_movie(necklace_base='W', k_start=3, k_end=18, depth=16, bar_max=2000)
# make_movie(necklace_base='BW', k_start=2, k_end=13, depth=14, bar_max=20000)
# make_movie(necklace_base='WB', k_start=2, k_end=13, depth=14, bar_max=20000)

make_movie(necklace_base='BBW', k_start=2, k_end=13, depth=14, bar_max=None)

plt.show()

