import math
import pickle
import numpy as np
from copy import deepcopy
from itertools import chain, combinations
from random import seed, uniform

seed('A string to use as a seed that allows us to reproduce the same network each time')
num_clonotypes = 5  # Number of clonotypes
num_peptides = 30  # Number of peptides
birth_rate_value = 2.5
death_rate_value = 1.0
peptide_stimulus_value = 150.0
peptide_prob_value = 6/30  # Probability that a peptide will be recognised by a random clonotype
realisations = 1  # Number of realisations
sharing_factor = 1.5

priming_start = 0.5
priming_end = 0.6
challenge_start = 1.1
challenge_end = 1.2


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

    def add_clonotype(self, clonotype):
        """
        Adds a clonotype to the list of clonotypes that recognise the peptide and increases *recognised* by 1.

        Parameters
        ----------
        clonotype : Clonotype
            Clonotype to be added.
        """

        if clonotype.position not in self.clonotypes:
            self.clonotypes.append(clonotype.position)
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
            cells += clonotype_list[i].cells

        return self.stimulus / cells


class Clonotype:
    """Class to represent clonotypes"""

    def __init__(self, num_cells, position_value, br_value, dr_value):
        """
        Parameters
        ----------
        num_cells : int
            Number of initial cells.
        position_value : int
            Number of the clonotype.
        br_value : float
            Birth rate for a cell in homeostasis
        dr_value : float
            Death rate for a cell.
        """

        self.naive = num_cells
        self.effector = 0
        self.memory = 0
        self.position = position_value
        self.br = br_value
        self.dr = dr_value
        self.peptides = []
        self.recognised = 0

    def sum_cells(self):
        """
        Sum the cells of the clonotype in all compartments.

        Returns
        -------
        int
            Total number of cells
        """
        return self.naive + self.effector + self.memory

    def add_peptide(self, peptide_list, subset=None):
        """
        Checks if peptides in *peptide_list* will be recognised by the clonotype.

        If *subset* is included checks if the peptides in the subset will be recognised with an increased probability.

        Parameters
        ----------
        peptide_list : list[Peptide]
            Peptides considered.
        subset : list[int]
            Subset of peptides with increased recognition probability.
        """

        if subset is None:
            for peptide in peptide_list:
                if peptide.position not in self.peptides:
                    check_value = uniform(0.0, 1.0)
                    if check_value < peptide.probability:
                        self.peptides.append(peptide.position)
                        self.recognised += 1
                        peptide.add_clonotype(self)
        elif len(subset) > 0:
            for index in subset:
                if peptide_list[index].position not in self.peptides:
                    check_value = uniform(0.0, 1.0)
                    if check_value < sharing_factor * peptide_list[index].probability:
                        self.peptides.append(peptide_list[index].position)
                        self.recognised += 1
                        peptide_list[index].add_clonotype(self)

    def birth_rate(self, clonotype_list=None, peptide_list=None):
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
            Birth rate of a cell in the clonotype.
        """

        if clonotype_list is None and peptide_list is None:
            return self.cells * self.br

        if clonotype_list is not None and peptide_list is not None:
            infection = 0.0
            for peptide in peptide_list:
                if peptide.position in self.peptides:
                    infection += peptide.cell_stimulus(clonotype_list)

            return self.cells * (self.br + infection)

    def death_rate(self):
        """
        Calculates the rate at which a cell of the clonotype dies.

        Returns
        -------
        float
            Death rate of a cell in the clonotype.
        """
        
        return self.cells * self.dr

    def death_rate_effector(self):
        """
        TEST FUNCTION
        """

        return 15 * self.cells * self.dr

    def birth(self):
        self.cells += 1

    def death(self):
        self.cells -= 1

    def n_death_check(self, num_deaths):
        if self.cells < num_deaths:
            return False
        else:
            return True

    def n_death(self, num_deaths):
        self.cells -= num_deaths


initial_clones = [Clonotype(5, i, birth_rate_value, death_rate_value) for i in range(num_clonotypes)]
peptides = [Peptide(peptide_prob_value, i, peptide_stimulus_value) for i in range(num_peptides)]

for clone in initial_clones:
    clone.add_peptide(peptides)

seed()

states = []
times = []

for current_realisation in range(realisations):

    clones = deepcopy(initial_clones)

    realisation_states = [deepcopy([clone.cells for clone in clones])]
    realisation_times = [0.0]

    current_time = 0.0

    # Gillespie algorithm
    # Pre-infection stage
    while True:
        r1 = uniform(0.0, 1.0)
        r2 = uniform(0.0, 1.0)

        alpha = np.array([])
        for clone in clones:
            alpha = np.append(alpha, clone.birth_rate())
        for clone in clones:
            alpha = np.append(alpha, clone.death_rate())

        alpha_sum = alpha.sum()

        dt = -math.log(r1) / alpha_sum
        current_time += dt

        if current_time >= priming_start:
            realisation_states.append(deepcopy([clone.cells for clone in clones]))
            realisation_times.append(deepcopy(priming_start))
            break

        for current_rate in range(alpha.size):
            if (sum(alpha[:current_rate]) / alpha_sum) <= r2 < (sum(alpha[:current_rate + 1]) / alpha_sum):
                current_event = int(current_rate / num_clonotypes)
                current_clone = current_rate % num_clonotypes

                if current_event == 0:  # Birth event
                    clones[current_clone].birth()
                if current_event == 1:  # Death event
                    clones[current_clone].death()

        realisation_states.append(deepcopy([clone.cells for clone in clones]))
        realisation_times.append(deepcopy(current_time))

    # Priming stage
    while True:
        r1 = uniform(0.0, 1.0)
        r2 = uniform(0.0, 1.0)

        alpha = np.array([])
        for clone in clones:
            alpha = np.append(alpha, clone.birth_rate(clones, peptides[:int(len(peptides) / 2)]))
        for clone in clones:
            alpha = np.append(alpha, clone.death_rate())

        alpha_sum = alpha.sum()

        dt = -math.log(r1) / alpha_sum
        current_time += dt

        if current_time >= priming_end:
            realisation_states.append(deepcopy([clone.cells for clone in clones]))
            realisation_times.append(deepcopy(priming_end))
            break

        for current_rate in range(alpha.size):
            if (sum(alpha[:current_rate]) / alpha_sum) <= r2 < (sum(alpha[:current_rate + 1]) / alpha_sum):
                current_event = int(current_rate / num_clonotypes)
                current_clone = current_rate % num_clonotypes

                if current_event == 0:  # Birth event
                    clones[current_clone].birth()
                if current_event == 1:  # Death event
                    clones[current_clone].death()

        realisation_states.append(deepcopy([clone.cells for clone in clones]))
        realisation_times.append(deepcopy(current_time))

    # Memory stage
    while True:
        r1 = uniform(0.0, 1.0)
        r2 = uniform(0.0, 1.0)

        alpha = np.array([])
        for clone in clones:
            alpha = np.append(alpha, clone.birth_rate())
        for clone in clones:
            if current_time < priming_end + 0.1:
                alpha = np.append(alpha, clone.death_rate_effector())
            else:
                alpha = np.append(alpha, clone.death_rate())
            # alpha = np.append(alpha, clone.death_rate())

        alpha_sum = alpha.sum()

        dt = -math.log(r1) / alpha_sum
        current_time += dt

        if current_time >= challenge_start:
            realisation_states.append(deepcopy([clone.cells for clone in clones]))
            realisation_times.append(deepcopy(challenge_start))
            break

        for current_rate in range(alpha.size):
            if (sum(alpha[:current_rate]) / alpha_sum) <= r2 < (sum(alpha[:current_rate + 1]) / alpha_sum):
                current_event = int(current_rate / num_clonotypes)
                current_clone = current_rate % num_clonotypes

                if current_event == 0:  # Birth event
                    clones[current_clone].birth()
                if current_event == 1:  # Death event
                    clones[current_clone].death()

        realisation_states.append(deepcopy([clone.cells for clone in clones]))
        realisation_times.append(deepcopy(current_time))

    # Challenge stage
    while True:
        r1 = uniform(0.0, 1.0)
        r2 = uniform(0.0, 1.0)

        alpha = np.array([])
        for clone in clones:
            alpha = np.append(alpha, clone.birth_rate(clones, peptides[int(len(peptides) / 2):]))
        for clone in clones:
            alpha = np.append(alpha, clone.death_rate())

        alpha_sum = alpha.sum()

        dt = -math.log(r1) / alpha_sum
        current_time += dt

        if current_time >= challenge_end:
            realisation_states.append(deepcopy([clone.cells for clone in clones]))
            realisation_times.append(deepcopy(challenge_end))
            break

        for current_rate in range(alpha.size):
            if (sum(alpha[:current_rate]) / alpha_sum) <= r2 < (sum(alpha[:current_rate + 1]) / alpha_sum):
                current_event = int(current_rate / num_clonotypes)
                current_clone = current_rate % num_clonotypes

                if current_event == 0:  # Birth event
                    clones[current_clone].birth()
                if current_event == 1:  # Death event
                    clones[current_clone].death()

        realisation_states.append(deepcopy([clone.cells for clone in clones]))
        realisation_times.append(deepcopy(current_time))

    # Post-infection stage
    while current_time < 1.5:
        r1 = uniform(0.0, 1.0)
        r2 = uniform(0.0, 1.0)

        alpha = np.array([])
        for clone in clones:
            alpha = np.append(alpha, clone.birth_rate())
        for clone in clones:
            if current_time < challenge_end + 0.1:
                alpha = np.append(alpha, clone.death_rate_effector())
            else:
                alpha = np.append(alpha, clone.death_rate())

        alpha_sum = alpha.sum()

        dt = -math.log(r1) / alpha_sum
        current_time += dt

        for current_rate in range(alpha.size):
            if (sum(alpha[:current_rate]) / alpha_sum) <= r2 < (sum(alpha[:current_rate + 1]) / alpha_sum):
                current_event = int(current_rate / num_clonotypes)
                current_clone = current_rate % num_clonotypes

                if current_event == 0:  # Birth event
                    clones[current_clone].birth()
                if current_event == 1:  # Death event
                    clones[current_clone].death()

        realisation_states.append(deepcopy([clone.cells for clone in clones]))
        realisation_times.append(deepcopy(current_time))

    states.append(deepcopy(realisation_states))
    times.append(deepcopy(realisation_times))

with open('Data.bin', 'wb') as file:
    data = (states, times)
    pickle.dump(data, file)
