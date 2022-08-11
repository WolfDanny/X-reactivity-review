# %% Packages


from adaptive import *
from pylab import savefig, step
from distutils.spawn import find_executable
import pickle
import matplotlib.pyplot as plt
import networkx as nx

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

# %% Loading results


with open("Data.bin", "rb") as file:
    data = pickle.load(file)

    clone_states = data[0]
    result_times = data[1]
    priming_start = data[2]
    priming_end = data[3]
    challenge_start = data[4]
    challenge_end = data[5]
    experiment_end = data[6]
    network = data[7]
    clonotypes = data[8]
    peptides = data[9]
    naive = data[10]
    effector = data[11]
    memory = data[12]

    del data

# %% Generating plots


times = []
clones = [[], [], [], [], []]
clones_naive = [[], [], [], [], []]
clones_effector = [[], [], [], [], []]
clones_memory = [[], [], [], [], []]

ticks = np.arange(0, experiment_end, 1 / 12) + 1 / 12
labels = np.arange(0, len(ticks), 1) + 1

for t in result_times[0]:
    times.append(t)

for state in clone_states[0]:
    clones[0].append(state[0])
    clones[1].append(state[1])
    clones[2].append(state[2])

for state in naive[0]:
    clones_naive[0].append(state[0])
    clones_naive[1].append(state[1])
    clones_naive[2].append(state[2])

for state in effector[0]:
    clones_effector[0].append(state[0])
    clones_effector[1].append(state[1])
    clones_effector[2].append(state[2])

for state in memory[0]:
    clones_memory[0].append(state[0])
    clones_memory[1].append(state[1])
    clones_memory[2].append(state[2])

ratio = 12 / 4.5
height = 5
figs, graphs = plt.subplots(
    constrained_layout=False, figsize=(ratio * height, height), tight_layout=True
)

if network == 0:
    network_name = "U"
elif network == 1:
    network_name = "D"
elif network == 2:
    network_name = "A"
else:
    network_name = "Error"

G = nx.Graph()
G.add_nodes_from(["C{}".format(i + 1) for i in range(3)], bipartite=0)

G.add_nodes_from([i + 1 for i in range(9)], bipartite=1)
G.add_nodes_from([i + 1 for i in range(9, 18)], bipartite=2)

edges = []
for clone_number in range(len(clonotypes)):
    for vdp in clonotypes[clone_number].peptides:
        edges.append(("C{}".format(clone_number + 1), vdp + 1))
G.add_edges_from(edges)

nodes = G.nodes()

clonotype_nodes = set([n for n in nodes if G.nodes[n]["bipartite"] == 0])
left_peptides = set([n for n in nodes if G.nodes[n]["bipartite"] == 1])
right_peptides = set([n for n in nodes if G.nodes[n]["bipartite"] == 2])

pos = dict()
clone_scale = 2
centering_factor = (
    max(len(left_peptides), len(right_peptides))
    - (len(clonotype_nodes) + ((len(clonotype_nodes) - 1) * (clone_scale - 1)))
) / 2

pos.update((n, (1, i)) for i, n in enumerate(sorted(left_peptides)))
pos.update((n, (3, i)) for i, n in enumerate(sorted(right_peptides)))
pos.update(
    (n, (2, (clone_scale * i) + centering_factor))
    for i, n in enumerate(sorted(clonotype_nodes))
)

colour = ["yellowgreen", "lightseagreen", "mediumorchid"]

plt.subplot(131)

options = {"node_size": clone_scale * 600}
nx.draw_networkx_nodes(
    G,
    pos=pos,
    nodelist=sorted(clonotype_nodes),
    node_color=colour,
    node_shape="o",
    **options
)
nx.draw_networkx_nodes(
    G, pos=pos, nodelist=left_peptides, node_color="blue", node_shape="D"
)
nx.draw_networkx_nodes(
    G, pos=pos, nodelist=right_peptides, node_color="orangered", node_shape="D"
)
nx.draw_networkx_edges(G, pos=pos)

net_labels = {"C1": r"$C_{1}$", "C2": r"$C_{2}$", "C3": r"$C_{3}$"}
nx.draw_networkx_labels(
    G, pos=pos, labels=net_labels, font_size=18 + (2 * (clone_scale - 1))
)

plt.axis("off")

combined_plot = plt.subplot(132)

step(times, [n for n in clones[0]], "-", label="$C_1$", where="post", color=colour[0])
step(times, [n for n in clones[1]], "-", label="$C_2$", where="post", color=colour[1])
step(times, [n for n in clones[2]], "-", label="$C_3$", where="post", color=colour[2])

combined_plot.axvspan(priming_start, priming_end, alpha=0.15, color="black")
combined_plot.axvspan(challenge_start, challenge_end, alpha=0.15, color="black")

total_max = 90
naive_max = 6
memory_max = 13

combined_plot.set_ylabel("Number of cells", fontsize=label_size)
combined_plot.set_xlabel("$t$ / months", fontsize=label_size)
combined_plot.set_xlim(0, experiment_end)
combined_plot.set_ylim(0, total_max)
combined_plot.set_xticks(ticks)
combined_plot.set_xticklabels(labels)
combined_plot.legend(loc="upper right", fontsize="x-large")

for plotted_clone in range(3):

    naive_plot = plt.subplot(333)
    step(
        times,
        [n for n in clones_naive[plotted_clone]],
        "-",
        label="$C_{}$ naive".format(plotted_clone + 1),
        where="post",
        color=colour[plotted_clone],
    )
    naive_plot.axvspan(priming_start, priming_end, alpha=0.15, color="black")
    naive_plot.axvspan(challenge_start, challenge_end, alpha=0.15, color="black")
    naive_plot.set_xlim(0, experiment_end)
    naive_plot.set_ylim(0, naive_max)
    naive_plot.set_xticks(ticks)
    plt.setp(naive_plot.get_xticklabels(), visible=False)
    naive_plot.set_title(
        "$C_{}$ naive cells".format(plotted_clone + 1), fontsize=title_size
    )

    effector_plot = plt.subplot(336, sharex=naive_plot)
    step(
        times,
        [n for n in clones_effector[plotted_clone]],
        "-",
        label="$C_{}$ effector".format(plotted_clone + 1),
        where="post",
        color=colour[plotted_clone],
    )
    effector_plot.axvspan(priming_start, priming_end, alpha=0.15, color="black")
    effector_plot.axvspan(challenge_start, challenge_end, alpha=0.15, color="black")
    effector_plot.set_xlim(0, experiment_end)
    effector_plot.set_ylim(0, total_max)
    effector_plot.set_xticks(ticks)
    plt.setp(effector_plot.get_xticklabels(), visible=False)
    effector_plot.set_title(
        "$C_{}$ effector cells".format(plotted_clone + 1), fontsize=title_size
    )

    memory_plot = plt.subplot(339, sharex=naive_plot)
    step(
        times,
        [n for n in clones_memory[plotted_clone]],
        "-",
        label="$C_{}$ memory".format(plotted_clone + 1),
        where="post",
        color=colour[plotted_clone],
    )
    memory_plot.axvspan(priming_start, priming_end, alpha=0.15, color="black")
    memory_plot.axvspan(challenge_start, challenge_end, alpha=0.15, color="black")
    memory_plot.set_xlim(0, experiment_end)
    memory_plot.set_ylim(0, memory_max)
    memory_plot.set_xticks(ticks)
    memory_plot.set_xticklabels(labels)
    memory_plot.set_title(
        "$C_{}$ memory cells".format(plotted_clone + 1), fontsize=title_size
    )
    memory_plot.set_xlabel("$t$ / months", fontsize=label_size)

    savefig("Graph-{0}-C{1}.pdf".format(network_name, plotted_clone + 1))
    naive_plot.clear()
    effector_plot.clear()
    memory_plot.clear()

plt.close(fig="all")
