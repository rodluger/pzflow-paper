"""Create the two moons example figure."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from pzflow import Flow
from pzflow.examples import get_twomoons_data

# load the two moons example data
data = get_twomoons_data()

# create and train a flow to model the two moons data
flow = Flow(data.columns)
_ = flow.train(data)

# calculate the latent variables for a subset of the data
samples = data[:5_000].to_numpy()
latents = flow._forward(
    flow._params[1], samples, conditions=np.zeros((samples.shape[0], 1))
)[0]


# set the colors of the samples
def thread(x: np.ndarray) -> np.ndarray:
    """Calculate function that threads between the two moons."""
    return 0.5 * np.sin(-3 * (x - 0.5)) + 0.25


colors = []
for row in samples:
    # upper moon
    if thread(row[0]) < row[1]:
        # left half
        if row[0] < 0:
            colors.append("C0")
        # right half
        else:
            colors.append("C1")

    # lower moon
    else:
        # left half
        if row[0] > 1:
            colors.append("C2")
        # right half
        else:
            colors.append("C3")


# create the figure
fig = plt.figure(figsize=(7.1, 3), constrained_layout=True)
gs = GridSpec(1, 5, figure=fig)

# make the scatter plots
scatter_settings = {
    "c": colors,
    "marker": ".",
    "s": 3,
    "rasterized": True,
}

# scatter plot of data (left panel)
ax1 = fig.add_subplot(gs[:2])
ax1.scatter(samples[:, 0], samples[:, 1], **scatter_settings)
ax1.set(
    title="Data Space",
    xlim=(-1.5, 2.5),
    ylim=(-1, 1.5),
    xticks=[],
    yticks=[],
    aspect=4 / 2.5,
)

# scatter plot of latents (right panel)
ax2 = fig.add_subplot(gs[3:])
ax2.scatter(latents[:, 0], latents[:, 1], **scatter_settings)
ax2.set(
    title="Latent Space",
    xlim=(-5.5, 5.5),
    ylim=(-5.5, 5.5),
    xticks=[],
    yticks=[],
    aspect=1,
)

# add the text in between
center = fig.add_subplot(gs[2])
center.axis("off")  # hide the axes

# right arrow and text
right_arrow = "$" + "\!\!\!\!".join(10 * ["-"] + ["\longrightarrow"]) + "$"
center.text(
    0.5,
    0.55,
    "$x \sim p_X$\n$u = f(x)$\n" + right_arrow,
    transform=center.transAxes,
    ha="center",
    va="bottom",
)

# left arrow and text
left_arrow = "$" + "\!\!\!\!".join(["\longleftarrow"] + 10 * ["-"]) + "$"
center.text(
    0.5,
    0.45,
    left_arrow + "\n$u \sim p_U$\n$x = f^{\,-1}(u)$",
    transform=center.transAxes,
    ha="center",
    va="top",
)

# save the figure!
fig.savefig("twomoons_example.pdf", dpi=600)
