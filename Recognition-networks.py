import math
import pickle
import numpy as np
from copy import deepcopy
from itertools import chain, combinations
from random import uniform

num_clonotypes = 5  # Number of clonotypes
num_peptides = 30  # Number of peptides
peptide_prob_value = 5 / 30  # Probability that a peptide will be recognised by a random clonotype


class Peptide:
    """Class for peptide objects"""

    def __init__(self, probability_value, position_value):
        """
        Parameters
        ----------
        probability_value : float
            Probability of peptide being recognised
        position_value : int
            Number of the peptide.
        """

        self.probability = probability_value
        self.position = position_value
        self.clonotypes = []
        self.recognised = 0

    def add_clonotype(self, clonotype):
        """
        Adds a clonotype to the list of clonotypes that recognise the peptide, and increases *recognised* by 1.

        Parameters
        ----------
        clonotype : Clonotype
            Clonotype to be added.
        """

        if clonotype.position not in self.clonotypes:
            self.clonotypes.append(clonotype.position)
            self.recognised += 1


class Clonotype:
    """Class for clonotype objects"""

    def __init__(self, num_cells, position_value):
        """
        Parameters
        ----------
        num_cells : int
            Number of initial cells.
        position_value : int
            Number of the clonotype.
        """

        self.cells = num_cells
        self.position = position_value
        self.peptides = []

    def add_peptide(self, peptide_list, subset=None):
        """
        Checks if peptides in *peptide_list* will be recognised by the clonotype,
        and appends them to *peptides* if they are.

        Parameters
        ----------
        peptide_list : list[Peptide]
            Peptide considered.
        subset : list[int]
            Subset of
        """

        if subset is None:
            for peptide in peptide_list:
                if peptide.position not in self.peptides:
                    check_value = uniform(0.0, 1.0)
                    if check_value < peptide.probability:
                        self.peptides.append(peptide.position)
                        peptide.add_clonotype(self)

    def birth(self):
        self.cells += 1

    def n_death_check(self, num_deaths):
        if self.cells < num_deaths:
            return False
        else:
            return True

    def n_death(self, num_deaths):
        self.cells -= num_deaths
