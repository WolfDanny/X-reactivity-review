# %% Packages


from distutils.spawn import find_executable
import matplotlib.pyplot as plt
import numpy as np
import crossreactivity.definitions as xr

# %% Definitions


if find_executable("latex"):
    plt.rcParams.update({"text.usetex": True})
    legend_size = 14
    label_size = 18
    title_size = 22
else:
    legend_size = 12
    label_size = 16
    title_size = 18
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# %% Parameters


recognition_probabilities = np.linspace(0, 1, 100)
clonotypes = 8
epitopes = 20
variations = 5
crossreactivities = np.linspace(0.01, 0.3, variations)
mmsb_epitopes = [
    [
        xr.epitope_recognition_vector(epitopes, blocks, probability / 2, probability)
        for probability in recognition_probabilities
    ]
    for blocks in list(range(1, variations + 1))
]
mmsb_clonotypes = [
    xr.affiliation_vectors(clonotypes, epitopes, blocks)
    for blocks in list(range(2, variations + 1))
]

# %% Generating data


er_data = [
    xr.clustering_coefficient(
        "er", recognition_probability=probability, clonotypes=clonotypes
    )
    for probability in recognition_probabilities
]
c_data = [
    xr.clustering_coefficient(
        "c",
        recognition_probability=probability,
        clonotypes=clonotypes,
        epitopes=epitopes,
    )
    for probability in recognition_probabilities
]
pa_data = [
    [
        xr.clustering_coefficient(
            "pa",
            recognition_probability=probability,
            clonotypes=clonotypes,
            epitopes=epitopes,
            crossreactivity=crossreactivity,
        )
        for probability in recognition_probabilities
    ]
    for crossreactivity in crossreactivities
]
mmsb_data_single_probability = [
    [
        xr.clustering_coefficient(
            "mmsb",
            clonotype_probabilities=probabilities,
            epitope_probabilities=epitope_probabilities,
        )
        for epitope_probabilities in mmsb_epitopes[0]
    ]
    for probabilities in mmsb_clonotypes
]
mmsb_data_block_probabilities = [
    [
        xr.clustering_coefficient(
            "mmsb",
            clonotype_probabilities=mmsb_clonotypes[i - 1],
            epitope_probabilities=epitope_probabilities,
        )
        for epitope_probabilities in mmsb_epitopes[i]
    ]
    for i in range(1, variations)
]

# %% Generating figures


h = 5
lw = 1.5
patterns = ["-", (0, (6, 4)), (0, (7, 5, 2, 5)), (0, (2, 4)), (0, (1, 1))]

fig, graph = plt.subplots(1, 1, constrained_layout=True, figsize=(h, h))

for data, linestyle, label in zip(
    [er_data, c_data, pa_data[-1], mmsb_data_single_probability[0]],
    patterns,
    [
        "Erd{\\H{o}}s-R\\'enyi",
        "Configuration model",
        "Preferential attachment",
        "Stochastic blockmodel",
    ],
):
    graph.plot(recognition_probabilities, data, lw=lw, linestyle=linestyle, label=label)
graph.set_title("Random recognition networks", fontsize=title_size)
graph.set_xlabel("$p_{v}$", fontsize=label_size)
graph.set_ylabel("$\\hat{C}_{4}$", fontsize=label_size)
graph.set_facecolor("white")
graph.set_ylim(0, 1)
graph.set_xlim(0, 1)
graph.set_aspect("equal")
graph.tick_params(axis="both", labelsize=11)
graph.legend(loc=2, fontsize=legend_size)

fig.savefig("Combined.pdf")
plt.close(fig="all")

fig, graph = plt.subplots(1, 1, constrained_layout=True, figsize=(h, h))

for data, linestyle, label in zip(
    pa_data,
    patterns,
    [f"$p^{{*}}_{{v}}={round(value, 3)}$" for value in crossreactivities],
):
    graph.plot(recognition_probabilities, data, lw=lw, linestyle=linestyle, label=label)
graph.set_title("Preferential attachment networks", fontsize=title_size)
graph.set_xlabel("$p_{v}$", fontsize=label_size)
graph.set_ylabel("$\\hat{C}_{4}$", fontsize=label_size)
graph.set_facecolor("white")
graph.set_ylim(0, 1)
graph.set_xlim(0, 1)
graph.set_aspect("equal")
graph.tick_params(axis="both", labelsize=11)
graph.legend(loc=2, fontsize=legend_size)

fig.savefig("PA.pdf")
plt.close(fig="all")
