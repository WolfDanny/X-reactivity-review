import math
import pickle
import numpy as np
from copy import deepcopy
from itertools import chain, combinations
from random import seed, uniform

seed('A random string to generate a seed that allows us to reproduce the same network each time')
num_clonotypes = 4  # Number of clonotypes
num_peptides = 10  # Number of peptides
birth_rate_value = 1.0
peptide_stimulus_value = 10.0
peptide_prob_value = 2/5  # Probability that a peptide will be recognised by a random clonotype
realisations = 1  # Number of realisations
sharing_factor = 1.5


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

    def __init__(self, num_cells, position_value, br_value):
        """
        Parameters
        ----------
        num_cells : int
            Number of initial cells.
        position_value : int
            Number of the clonotype.
        br_value : float
            Birth rate for a cell in homeostasis
        """

        self.cells = num_cells
        self.position = position_value
        self.homeostatic_br = br_value
        self.peptides = []
        self.recognised = 0

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

    def birth_rate(self, clonotype_list, peptide_list):
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
            Birth rate for clone in state.
        """

        foreign = 0.0
        for peptide in peptide_list:
            if peptide.position in self.peptides:
                foreign += peptide.cell_stimulus(clonotype_list)

        return self.cells * (self.homeostatic_br + foreign)

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


clones = [Clonotype(5, i) for i in range(num_clonotypes)]
peptides = [Peptide(peptide_prob_value, i) for i in range(num_peptides)]

for clone in clones:
    clone.add_peptide(peptides)

seed()

states = []
times = []

for current_realisation in range(realisations):
    current_states = [[clone.cells for clone in clones]]
    current_times = [0]

    # Gillespie stuff
    while 1 < 0:
        pass

    states.append(current_states)
    times.append(current_times)
