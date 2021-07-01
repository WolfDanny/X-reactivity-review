from pylab import savefig,step
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from itertools import chain, combinations
from random import uniform, sample
import numpy as np

plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

class Peptide:
    """Class to represent peptides"""

    def __init__(self, probability_value, position_value, stimulus_value):
        """
        Parameters
        ----------
        probability_value : float
            Probability of peptide being recognised.
        position_value : int
            Number of the peptide.
        stimulus_value : float
            Stimulus provided by the peptide.
        """

        self.probability = probability_value
        self.position = position_value
        self.stimulus = stimulus_value
        self.clonotypes = []
        self.recognised = 0

    def add_clonotype(self, clone):
        """
        Adds a clonotype to the list of clonotypes that recognise the peptide and increases *recognised* by 1.

        Parameters
        ----------
        clone : Clonotype
            Clonotype to be added.
        """

        if clone.position not in self.clonotypes:
            self.clonotypes.append(clone.position)
            self.recognised += 1

    def cell_stimulus(self, clonotype_list):
        """
        Calculates the stimulus provided to a cell that recognises the peptide.

        Parameters
        ----------
        clonotype_list : list[Clonotype]
            Current list of clonotypes.

        Returns
        -------
        float
            Stimulus provided to a cell that recognises the peptide
        """

        cells = 0

        for i in self.clonotypes:
            cells += clonotype_list[i].stimulated_cells()

        try:
            return self.stimulus / cells
        except ZeroDivisionError:
            return 0


class Clonotype:
    """Class to represent clonotypes"""

    def __init__(self, position, num_cells, naive_homeostatic_rate, naive_competition_matrix, memory_homeostatic_rate, effector_division_constant, naive_differentiation_constant, memory_differentiation_constant, naive_death_rate, effector_death_rate, memory_death_rate, effector_differentiation_rate):
        """
        Parameters
        ----------
        position : int
            Number of the clonotype.
        num_cells : int
            Number of initial cells.
        naive_homeostatic_rate : float
            Homeostatic proliferation rate for a naive cell.
        naive_competition_matrix : numpy.ndarray
            Homeostatic competition sharing probability matrix.
        memory_homeostatic_rate : float
            Birth rate for a memory cell.
        effector_division_constant : float
            Constant of effector division.
        naive_death_rate : float
            Death rate for a naive cell.
        effector_death_rate : float
            Death rate for an effector cell
        memory_death_rate : float
            Death rate for a memory cell.
        naive_differentiation_constant : float
            Differentiation constant from naive to effector
        memory_differentiation_constant : float
            Differentiation constant from memory to effector
        effector_differentiation_rate : float
            Differentiation rate from effector to memory.
        """

        self.naive = num_cells
        self.effector = 0
        self.effector_dividing = 0
        self.effector_division_times = []
        self.memory = 0
        self.position = position
        self.naive_homeostatic_rate = naive_homeostatic_rate
        self.naive_competition_matrix = naive_competition_matrix
        self.memory_homeostatic_rate = memory_homeostatic_rate
        self.effector_division_constant = effector_division_constant
        self.naive_death_rate = naive_death_rate
        self.memory_death_rate = memory_death_rate
        self.effector_death_rate = effector_death_rate
        self.naive_differentiation_constant = naive_differentiation_constant
        self.memory_differentiation_constant = memory_differentiation_constant
        self.effector_differentiation_rate = effector_differentiation_rate
        self.peptides = []
        self.recognised = 0

    def cells_list(self):
        """
        Creates a list with the populations of cells in the clonotype in the three compartments.

        Returns
        -------
        list[int]
            List of the current cell populations in the 3 compartments
        """

        return [self.naive, self.effector + self.effector_dividing, self.memory]

    def cells(self):
        """
        Returns the total number of cells of the clonotype.

        Returns
        -------
        int
            Total number of cells
        """
        return self.naive + self.effector + self.effector_dividing + self.memory

    def stimulated_cells(self):
        """
        Returns the total number of cells that can receive stimulus from peptides, that is it excludes effector cells that are already dividing.

        Returns
        -------
        int
            Total number of cells that can receive stimulus.
        """

        return self.naive + self.effector + self.memory

    def add_peptide(self, peptide_list, network_type, degree=None, subset=None, clone_list=None):
        """
        Checks if peptides in *peptide_list* will be recognised by the clonotype.

        The parameter *network_type* represents the type of recognition network being constructed, where 0 is an unfocussed network, 1 is a degree focussed network, and 2 is a preferential attachment focussed network.

        - For an unfocussed network no optional parameters are necessary.
        - For a degree focused network the *degree* parameter is necessary.
        - For a preferential attachment focussed network the *clone_list* parameter is necessary.

        Parameters
        ----------
        peptide_list : list[Peptide]
            Peptides considered.
        network_type: int
            Type of network constructed.
        degree : int
            Number of peptides a clonotype will recognise in the degree focussed network.
        subset : list[int]
            List of peptides to be be considered for recognition by a clonotype.
        clone_list : list[Clonotype]
            List of clonotypes currently in the network.
        """

        # Unfocussed network
        if network_type == 0:
            if subset is None:
                for peptide in peptide_list:
                    if peptide.position not in self.peptides:
                        check_value = uniform(0.0, 1.0)
                        if check_value < peptide.probability:
                            self.peptides.append(peptide.position)
                            self.recognised += 1
                            peptide.add_clonotype(self)
            else:
                for peptide_index in subset:
                    if peptide_index not in self.peptides:
                        check_value = uniform(0.0, 1.0)
                        if check_value < 2 * peptide_list[peptide_index].probability:
                            self.peptides.append(peptide_index)
                            self.recognised += 1
                            peptide_list[peptide_index].add_clonotype(self)

        # Focussed network (degree)
        if network_type == 1 and degree is not None:
            peptide_sample = sample(list(range(len(peptide_list))), degree)
            for peptide_index in peptide_sample:
                self.peptides.append(peptide_index)
                self.recognised += 1
                peptide_list[peptide_index].add_clonotype(self)

        # Focussed network (preferential attachment)
        if network_type == 2 and clone_list is not None:
            self.add_peptide(peptide_list, 0)
            if len(clone_list) > 1:
                extra_peptides = []
                for peptide in self.peptides:
                    for clone in peptide_list[peptide].clonotypes:
                        for peptide_index in clone_list[clone].peptides:
                            if peptide_index not in self.peptides and peptide_index not in extra_peptides:
                                extra_peptides.append(peptide_index)
                self.add_peptide(peptide_list, 0, subset=extra_peptides)

    def birth_rate_naive(self, clonotype_list):
        """
        Calculates the rate at which a new cell of the clonotype is born.

        Parameters
        ----------
        clonotype_list : list[Clonotype]
            Current list of clonotypes.

        Returns
        -------
        float
            Birth rate of a naive cell in the clonotype.
        """

        rate = 0.0
        sets = clone_sets(len(clonotype_list), self.position)

        for i in range(len(sets)):
            total_cells = 0
            for clone in sets[i]:
                total_cells += clonotype_list[clone].naive
            if total_cells != 0:
                rate += self.naive_competition_matrix[self.position][i] / total_cells

        return rate * self.naive * self.naive_homeostatic_rate

    def birth_rate_memory(self):
        """
        Calculates the rate at which a new cell of the clonotype is born.

        Returns
        -------
        float
            Birth rate of a memory cell in the clonotype.
        """

        return self.memory * self.memory_homeostatic_rate

    def birth_rate_effector(self, clonotype_list, peptide_list=None):
        """
        Calculates the rate at which a new cell of the clonotype is born.

        Parameters
        ----------
        clonotype_list : list[Clonotype]
            Current list of clonotypes.
        peptide_list : list[Peptide]
            Peptides considered.

        Returns
        -------
        float
            Birth rate of an effector cell in the clonotype.
        """

        if peptide_list is None:
            return 0.0

        if self.effector == 0:
            return 0.0

        infection = 0.0
        for peptide in peptide_list:
            if peptide.position in self.peptides:
                infection += peptide.cell_stimulus(clonotype_list)

        return self.effector * self.effector_division_constant * infection

    def death_rate_naive(self):
        """
        Calculates the death rate of a naive cell.

        Returns
        -------
        float
            Death rate of a naive cell in the clonotype.
        """

        return self.naive * self.naive_death_rate

    def death_rate_memory(self):
        """
        Calculates the death rate of a memory cell.

        Returns
        -------
        float
            Death rate of a memory cell in the clonotype.
        """

        return self.memory * self.memory_death_rate

    def death_rate_effector(self):
        """
        Calculates the death rate of an effector cell.

        Returns
        -------
        float
            Death rate of an effector cell in the clonotype.
        """

        return self.effector * self.effector_death_rate

    def differentiation_rate_ne(self, clonotype_list, peptide_list=None):
        """
        Calculates the differentiation rate from naive to effector.

        Parameters
        ----------
        clonotype_list : list[Clonotype]
            Current list of clonotypes.
        peptide_list : list[Peptide]
            Peptides considered.

        Returns
        -------
        float
            Differentiation rate from naive to effector.
        """

        if peptide_list is None:
            return 0.0

        if self.naive == 0:
            return 0.0

        infection = 0.0
        for peptide in peptide_list:
            if peptide.position in self.peptides:
                infection += peptide.cell_stimulus(clonotype_list)

        return self.naive * self.naive_differentiation_constant * infection

    def differentiation_rate_em(self, current_infection):
        """
        Calculates the differentiation rate from effector to memory.

        Parameters
        ----------
        current_infection : float
            Current stimulus rate available from infection.

        Returns
        -------
        float
            Differentiation rate from effector to memory.
        """

        return self.effector * (self.effector_differentiation_rate * np.exp(-current_infection))

    def differentiation_rate_me(self, clonotype_list, peptide_list=None):
        """
        Calculates the differentiation rate from memory to effector.

        Parameters
        ----------
        clonotype_list : list[Clonotype]
            Current list of clonotypes.
        peptide_list : list[Peptide]
            Peptides considered.

        Returns
        -------
        float
            Differentiation rate from naive to effector.
        """

        if peptide_list is None:
            return 0.0

        if self.memory == 0:
            return 0.0

        infection = 0.0
        for peptide in peptide_list:
            if peptide.position in self.peptides:
                infection += peptide.cell_stimulus(clonotype_list)

        return self.memory * self.memory_differentiation_constant * infection

    def birth(self, compartment, division_time):
        """
        Updates the population when a birth event occurs.

        Parameters
        ----------
        compartment : int
            Compartment in which the birth takes place.
        division_time : float
            Time at which the division process ends.
        """

        if compartment == 0:
            self.naive += 1
        if compartment == 1:
            self.effector -= 1
            self.effector_dividing += 1
            self.effector_division_times.append(division_time)
        if compartment == 2:
            self.memory += 1

    def effector_division(self):
        """
        Updates the population when an effector cells completes its division cycle and returns the time when this event happens.

        Returns
        -------
        float
            Time when the division cycle was completed.
        """

        self.effector_dividing -= 1
        self.effector += 2
        time = self.effector_division_times.pop(0)

        return time

    def death(self, compartment):
        """
        Updates the population when a death event occurs.

        Parameters
        ----------
        compartment : int
            Compartment in which the birth takes place
        """

        if compartment == 0:
            self.naive -= 1
        if compartment == 1:
            self.effector -= 1
        if compartment == 2:
            self.memory -= 1

    def differentiation(self, starting_compartment):
        """
        Updates the population when a differentiation event occurs.

        The parameter *starting_compartment* represents the starting compartment of the cell, where

        - 0 means the cell started in the naive compartment,
        - 1 means the cell started in the effector compartment,
        - 2 means the cell started in the memory compartment.

        Parameters
        ----------
        starting_compartment : int
            Starting compartment of the differentiating cell.
        """

        if starting_compartment == 0:
            self.naive -= 1
            self.effector += 1
        if starting_compartment == 1:
            self.effector -= 1
            self.memory += 1
        if starting_compartment == 2:
            self.effector += 1


def clone_sets(dimension, clone):
    """
    Creates an ordered list of tuples representing all subsets of a set of *dimension* elements that include the *clone*-th element.

    Parameters
    ----------
    dimension : int
        Number of elements.
    clone : int
        Specified element (starts at 0).

    Returns
    -------
    list[int]
        List of tuples representing all subsets of a set of *dimension* elements that include the *clone*-th element.
    """

    if clone >= dimension or clone < 0:
        return -1

    x = range(dimension)
    sets = list(chain(*[combinations(x, ni) for ni in range(dimension + 1)]))
    d = []

    for T in sets:
        if clone not in T:
            d.insert(0, sets.index(T))

    for i in d:
        sets.pop(i)

    return sets


with open('Data.bin', 'rb') as file:
    data = pickle.load(file)
        
    clone_states = data[0]
    times = data[1]
    priming_start = data[2]
    priming_end = data[3]
    challenge_start = data[4]
    challenge_end = data[5]
    network = data[6]
    clonotypes = data[7]
    peptides = data[8]
    naive = data[9]
    effector = data[10]
    memory = data[11]
        
    del data
    
time = []
clones = [[], [], [], [], []]
clones_naive = [[], [], [], [], []]
clones_effector = [[], [], [], [], []]
clones_memory = [[], [], [], [], []]

ticks = np.arange(0, 1, 1/12) + 1/12
labels = np.arange(0, 12, 1) + 1

for t in times[0]:
    time.append(t)

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
figs, graphs = plt.subplots(constrained_layout=False, figsize=(ratio * height, height), tight_layout=True)

if network == 0:
    network_type = 'U'
if network == 1:
    network_type = 'D'
if network == 2:
    network_type = 'A'

G = nx.Graph()
G.add_nodes_from(['C{}'.format(i + 1) for i in range(3)], bipartite=0)

G.add_nodes_from([i + 1 for i in range(9)], bipartite=1)
G.add_nodes_from([i + 1 for i in range(9, 18)], bipartite=2)

edges = []
for clone_number in range(len(clonotypes)):
    for peptide in clonotypes[clone_number].peptides:
        edges.append(('C{}'.format(clone_number + 1), peptide + 1))
G.add_edges_from(edges)

nodes = G.nodes()

clonotype_nodes = set([n for n in nodes if G.nodes[n]['bipartite'] == 0])
left_peptides = set([n for n in nodes if G.nodes[n]['bipartite'] == 1])
right_peptides = set([n for n in nodes if G.nodes[n]['bipartite'] == 2])

pos = dict()
centering_factor = (max(len(left_peptides), len(right_peptides)) - len(clonotype_nodes)) / 2

pos.update((n, (1, i)) for i, n in enumerate(sorted(left_peptides)))
pos.update((n, (3, i)) for i, n in enumerate(sorted(right_peptides)))
pos.update((n, (2, i + centering_factor)) for i, n in enumerate(sorted(clonotype_nodes)))

colour = ['yellowgreen', 'rosybrown', 'mediumorchid']

plt.subplot(131)

options = {"node_size": 600}
nx.draw_networkx_nodes(G, pos=pos, nodelist=sorted(clonotype_nodes), node_color=colour, node_shape='o', **options)
nx.draw_networkx_nodes(G, pos=pos, nodelist=left_peptides, node_color='blue', node_shape='D')
nx.draw_networkx_nodes(G, pos=pos, nodelist=right_peptides, node_color='orangered', node_shape='D')
nx.draw_networkx_edges(G, pos=pos)

net_labels = {'C1': r'$C_{1}$', 'C2': r'$C_{2}$', 'C3': r'$C_{3}$'}
nx.draw_networkx_labels(G, pos=pos, labels=net_labels, font_size=18)

plt.axis('off')

combined_plot = plt.subplot(132)

step(time, [n for n in clones[0]], '-', label='$C_1$', where='post', color=colour[0])
step(time, [n for n in clones[1]], '-', label='$C_2$', where='post', color=colour[1])
step(time, [n for n in clones[2]], '-', label='$C_3$', where='post', color=colour[2])

combined_plot.axvspan(priming_start, priming_end, alpha=0.15, color='black')
combined_plot.axvspan(challenge_start, challenge_end, alpha=0.15, color='black')

combined_plot.set_ylabel('Number of cells', fontsize=14)
combined_plot.set_xlabel('Time [Months]', fontsize=14)
combined_plot.set_xlim(0, 1)
combined_plot.set_ylim(ymin=0)
combined_plot.set_xticks(ticks)
combined_plot.set_xticklabels(labels)
combined_plot.legend(loc='upper right', fontsize='x-large')

for plotted_clone in range(3):

    naive_plot = plt.subplot(333)
    step(time, [n for n in clones_naive[plotted_clone]], '-', label='$C_{}$ naive'.format(plotted_clone + 1), where='post', color=colour[plotted_clone])
    naive_plot.axvspan(priming_start, priming_end, alpha=0.15, color='black')
    naive_plot.axvspan(challenge_start, challenge_end, alpha=0.15, color='black')
    naive_plot.set_xlim(0, 1)
    naive_plot.set_ylim(ymin=0)
    naive_plot.set_xticks(ticks)
    plt.setp(naive_plot.get_xticklabels(), visible=False)
    naive_plot.set_title('$C_{}$ naive cells'.format(plotted_clone + 1), fontsize=16)

    effector_plot = plt.subplot(336, sharex=naive_plot)
    step(time, [n for n in clones_effector[plotted_clone]], '-', label='$C_{}$ effector'.format(plotted_clone + 1), where='post', color=colour[plotted_clone])
    effector_plot.axvspan(priming_start, priming_end, alpha=0.15, color='black')
    effector_plot.axvspan(challenge_start, challenge_end, alpha=0.15, color='black')
    effector_plot.set_xlim(0, 1)
    effector_plot.set_ylim(ymin=0)
    effector_plot.set_xticks(ticks)
    plt.setp(effector_plot.get_xticklabels(), visible=False)
    effector_plot.set_title('$C_{}$ effector cells'.format(plotted_clone + 1), fontsize=16)

    memory_plot = plt.subplot(339, sharex=naive_plot)
    step(time, [n for n in clones_memory[plotted_clone]], '-', label='$C_{}$ memory'.format(plotted_clone + 1), where='post', color=colour[plotted_clone])
    memory_plot.axvspan(priming_start, priming_end, alpha=0.15, color='black')
    memory_plot.axvspan(challenge_start, challenge_end, alpha=0.15, color='black')
    memory_plot.set_xlim(0, 1)
    memory_plot.set_ylim(ymin=0)
    memory_plot.set_xticks(ticks)
    memory_plot.set_xticklabels(labels)
    memory_plot.set_title('$C_{}$ memory cells'.format(plotted_clone + 1), fontsize=16)
    memory_plot.set_xlabel('Time [Months]', fontsize=14)

    savefig('Graph-{0}-C{1}.pdf'.format(network_type, plotted_clone + 1))
    naive_plot.clear()
    effector_plot.clear()
    memory_plot.clear()

plt.close(fig='all')
