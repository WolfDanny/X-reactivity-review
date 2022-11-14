# %% Packages


from distutils.spawn import find_executable
import matplotlib.pyplot as plt
import numpy as np
import crossreactivity.definitions as xr

# %% Definitions


if find_executable("latex"):
    plt.rcParams.update({"text.usetex": True})
    label_size = 18
    title_size = 20
else:
    label_size = 16
    title_size = 18
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# %% Parameters

recognition_probabilities = np.linspace(0, 1, 30)
num_clonotypes = 8
num_epitopes = 20
