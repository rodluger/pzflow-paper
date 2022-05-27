"""Plot the training losses for the conditional galaxy flow."""
import matplotlib.pyplot as plt
import numpy as np
import paths

# load the losses
losses = np.load(paths.data / "conditional_galaxy_flow_losses.npy")

# plot the losses
fig, ax = plt.subplots(figsize=(3.2, 2.5), constrained_layout=True)
ax.plot(losses)
ax.set(xlabel="Epochs", ylabel="Loss")
fig.savefig(paths.figures / "conditional_galaxy_flow_losses.pdf")
