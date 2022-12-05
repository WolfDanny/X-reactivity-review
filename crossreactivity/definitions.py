# %% Packages

import math
import numpy as np
from typing import Union
from itertools import chain, combinations, permutations, cycle
from scipy.special import comb
from random import uniform, sample

# %% Definitions


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
        Adds a clonotype to the list of clonotypes that recognise the peptide and increases ``recognised`` by 1.

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

    def __init__(
        self,
        position,
        num_cells,
        naive_homeostatic_rate,
        naive_competition_matrix,
        memory_homeostatic_rate,
        effector_division_constant,
        naive_differentiation_constant,
        memory_differentiation_constant,
        naive_death_rate,
        effector_death_rate,
        memory_death_rate,
        effector_differentiation_rate,
    ):
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
        Returns the total number of cells that can receive stimulus from peptides, that is, it excludes effector cells that are already dividing.

        Returns
        -------
        int
            Total number of cells that can receive stimulus.
        """

        return self.naive + self.effector + self.memory

    def add_peptide(
        self, peptide_list, network_type, degree=None, subset=None, clone_list=None
    ):
        """
        Checks if peptides in ``peptide_list`` will be recognised by the clonotype.

        The parameter ``network_type`` represents the type of recognition network being constructed, where 0 is an unfocussed network, 1 is a degree focussed network, and 2 is a preferential attachment focussed network.

        - For an unfocussed network no optional parameters are necessary.
        - For a degree focused network the *degree* parameter is necessary.
        - For a preferential attachment focussed network the ``clone_list`` parameter is necessary.

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
                        if check_value < peptide_list[peptide_index].probability:
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
                            if (
                                peptide_index not in self.peptides
                                and peptide_index not in extra_peptides
                            ):
                                extra_peptides.append(peptide_index)
                self.add_peptide(peptide_list, 0, subset=extra_peptides)

    def birth_rate_naive(self, clonotype_list):
        """
        Calculates the rate at which a naive cell divides.

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
        Calculates the rate at which a memory cell divides.

        Returns
        -------
        float
            Birth rate of a memory cell in the clonotype.
        """

        return self.memory * self.memory_homeostatic_rate

    def birth_rate_effector(self, clonotype_list, peptide_list=None):
        """
        Calculates the rate at which an effector cell divides.

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

        if current_infection == 0:
            return self.effector * self.effector_differentiation_rate
        else:
            return 0

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
        Updates the population when an effector cell completes its division cycle and returns the time when this event happens.

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

        The parameter *compartment* represents the compartment of the cell, where

        - 0 means the cell is in the naive compartment,
        - 1 means the cell is in the effector compartment,
        - 2 means the cell is in the memory compartment.

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

        The parameter ``starting_compartment`` represents the starting compartment of the cell, where

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
            self.memory -= 1
            self.effector += 1


def clone_sets(dimension, clone):
    """
    Creates an ordered list of tuples representing all subsets of a set of ``dimension`` elements that include the *clone*-th element.

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


def gillespie_step(
    clone_list,
    time,
    division_time,
    current_infection,
    time_limit=None,
    peptide_list=None,
):
    """
    Performs a step of the Gillespie simulation and returns the updated state and time.

    If ``time_limit`` is included and the step makes ``time`` greater than ``time_limit`` returns the current state and ``time_limit``.

    Parameters
    ----------
    clone_list : list[Clonotype]
        List of clonotypes in the current state.
    time : float
        Current time.
    division_time : float
        Time for an effector cell to divide.
    current_infection : float
        Current stimulus rate available from infection.
    time_limit : float
        Maximum time simulated.
    peptide_list : list[Peptide]
        List of currently present peptides.

    Returns
    -------
    list[Clonotype]
        List of clonotypes in the updated state.
    float
        Updated time.
    """

    rate_list = np.array([])
    # Birth rates
    for clone in clone_list:
        rate_list = np.append(rate_list, clone.birth_rate_naive(clone_list))
    for clone in clone_list:
        rate_list = np.append(
            rate_list, clone.birth_rate_effector(clone_list, peptide_list)
        )
    for clone in clone_list:
        rate_list = np.append(rate_list, clone.birth_rate_memory())

    # Death rates
    for clone in clone_list:
        rate_list = np.append(rate_list, clone.death_rate_naive())
    for clone in clone_list:
        rate_list = np.append(rate_list, clone.death_rate_effector())
    for clone in clone_list:
        rate_list = np.append(rate_list, clone.death_rate_memory())

    # Differentiation rates
    for clone in clone_list:
        rate_list = np.append(
            rate_list, clone.differentiation_rate_ne(clone_list, peptide_list)
        )
    for clone in clone_list:
        rate_list = np.append(
            rate_list, clone.differentiation_rate_em(current_infection)
        )
    for clone in clone_list:
        rate_list = np.append(
            rate_list, clone.differentiation_rate_me(clone_list, peptide_list)
        )

    r1 = uniform(0.0, 1.0)
    r2 = uniform(0.0, 1.0)

    total_rate = rate_list.sum()

    dt = -math.log(r1) / total_rate
    time += dt

    if time_limit is not None and time >= time_limit:
        return clone_list, time_limit

    clones = []
    division_times = []
    for clone in range(len(clone_list)):
        for previous_division_time in clone_list[clone].effector_division_times:
            if previous_division_time <= time:
                clones.append(clone)
                division_times.append(previous_division_time)

    if len(clones) > 0:
        dividing_clone = clones[division_times.index(min(division_times))]
        time = clone_list[dividing_clone].effector_division()
        return clone_list, time

    clonotype_total = len(clone_list)

    for rate in range(rate_list.size):
        if (
            (sum(rate_list[:rate]) / total_rate)
            <= r2
            < (sum(rate_list[: rate + 1]) / total_rate)
        ):
            event = int(rate / (clonotype_total * 3))
            clone = rate % clonotype_total
            compartment = int(rate / clonotype_total) % 3

            if event == 0:  # Birth event
                clone_list[clone].birth(compartment, time + division_time)
            if event == 1:  # Death event
                clone_list[clone].death(compartment)
            if event == 2:  # Differentiation event
                clone_list[clone].differentiation(compartment)
            break

    return clone_list, time


def affiliation_vectors(clonotypes, epitopes, blocks):
    """
    Generates a list of affiliation vectors with different permutations of values.

    Parameters
    ----------
    clonotypes : int
        Number of clonotypes.
    epitopes : int
        Number of epitopes.
    blocks : int
        Number of blocks.

    Returns
    -------
    list[list[float]]
        Affiliation vectors for each clonotype.
    """

    affiliations = np.linspace(1, 0, blocks)
    vectors = [[] for _ in range(clonotypes)]
    block_sizes = [round(epitopes / blocks)] * blocks
    block_sizes[-1] -= sum(block_sizes) - epitopes

    for vector, affiliation in zip(vectors, cycle(permutations(affiliations))):
        for value, size in zip(affiliation, block_sizes):
            for _ in range(size):
                vector.append(value)
    return vectors


def epitope_recognition_vector(epitopes, blocks, p_min, p_max):
    """
    Generates a list of epitope recognition probabilities for an MMSB network.

    Parameters
    ----------
    epitopes : int
        Number of epitopes.
    blocks : int
        Number of blocks.
    p_min : float
        Minimum probability value.
    p_max : float
        Maximum probability value.

    Returns
    -------
    list[float]
        List of recognition probabilities
    """
    probabilities = np.linspace(p_max, p_min, blocks)
    vector = []
    block_sizes = [round(epitopes / blocks)] * blocks
    block_sizes[-1] -= sum(block_sizes) - epitopes

    for value, size in zip(probabilities, block_sizes):
        for _ in range(size):
            vector.append(value)
    return vector


def clustering_coefficient(network, **kwargs):
    """
    Calculates the clustering coefficient for the specified type of network.

    Parameters
    ----------
    network : str
        Type of network (MMRB, ER, C, or PA).
    kwargs :
        Network parameters for the different types of networks.

    Returns
    -------
    float
        Clustering coefficient for the specified network and parameters.
    """

    networks = {
        "MMSB": _mmsb,
        "ER": _erdos_renyi,
        "C": _configuration,
        "PA": _preferential_attachment,
    }

    try:
        return networks[network.upper()](**kwargs)
    except KeyError:
        return -1


def _mmsb(clonotype_probabilities, epitope_probabilities, **kwargs):
    """
    Calculates the clustering coefficient of a mixed membership stochastic blockmodel network.

    Parameters
    ----------
    clonotype_probabilities : list[list[float]]
        List of clonotype affiliation vectors.
    epitope_probabilities : list[float]
        List of epitope recognition probabilities.

    Returns
    -------
    float
        Clustering coefficient of a mixed membership stochastic blockmodel network.
    """

    num_clonotypes = len(clonotype_probabilities)
    num_epitopes = len(epitope_probabilities)

    value = 0
    for clone, _ in enumerate(clonotype_probabilities):
        for vdp_0, vdp_1 in combinations(list(range(num_epitopes)), 2):
            value += _local_mmsb(
                clone, vdp_0, vdp_1, clonotype_probabilities, epitope_probabilities
            )
    return value / (num_clonotypes * comb(num_epitopes, 2))


def _local_mmsb(clone, vdp_0, vdp_1, clonotype_probabilities, epitope_probabilities):
    """
    Calculates the local clustering coefficient for ``clone``, ``vdp_0``, and ``vdp_1`` in a mixed membership stochastic blockmodel network.

    Parameters
    ----------
    clone : int
        Index of the clonotype considered.
    vdp_0 : int
        Index of the first VDP considered.
    vdp_1 : int
        Index of the second VDP considered.
    clonotype_probabilities : list[list[float]]
        List of clonotype affiliation vectors.
    epitope_probabilities : list[float]
        List of epitope recognition probabilities.

    Returns
    -------
    float
        Local clustering coefficient for clone, vdp_0, and vdp_1.
    """
    epitope_degrees = []
    for index in [vdp_0, vdp_1]:
        value = 0
        for clonotype in clonotype_probabilities:
            value += epitope_probabilities[index] * clonotype[index]
        epitope_degrees.append(value)

    butterflies = (
        ((epitope_probabilities[vdp_0] * epitope_probabilities[vdp_1]) ** 2)
        * clonotype_probabilities[clone][vdp_0]
        * clonotype_probabilities[clone][vdp_1]
    )
    value = 0
    for index, probability in enumerate(clonotype_probabilities):
        if index != clone:
            value += probability[vdp_0] * probability[vdp_1]
    butterflies *= value

    return butterflies / (sum(epitope_degrees) - 2 - butterflies)


def _erdos_renyi(recognition_probability, clonotypes, **kwargs):
    """
    Calculates the clustering coefficient of an Erdos-Renyi network.

    Parameters
    ----------
    recognition_probability : float
        VDP recognition probability.
    clonotypes : int
        Number of clonotypes.

    Returns
    -------
    float
        Clustering coefficient of an Erdos-Renyi network.
    """

    return ((recognition_probability**4) * (clonotypes - 1)) / (
        (2 * recognition_probability * clonotypes)
        - 2
        - ((recognition_probability**4) * (clonotypes - 1))
    )


def _epitope_degrees(recognition_probability, clonotypes, epitopes):
    """
    Generates a constant list of epitope node degrees such that p_{v} is approximately ``probability``.

    Parameters
    ----------
    recognition_probability : float
        Epitope recognition probability.
    clonotypes : int
        Number of clonotypes.
    epitopes : int
        Number of epitopes.

    Returns
    -------
    list[int]
        List of degrees of epitope nodes.
    """
    return [round(recognition_probability * clonotypes)] * epitopes


def _configuration(recognition_probability, clonotypes, epitopes, **kwargs):
    """
    Calculates the clustering coefficient of a configuration model network.

    Parameters
    ----------
    recognition_probability : float
        Epitope recognition probability.
    clonotypes : int
        Number of clonotypes.
    epitopes : Union[int, list[int]]
        Number of epitopes or list of epitope degrees.

    Returns
    -------
    float
        Clustering coefficient of a configuration model network.
    """

    if isinstance(epitopes, int):
        epitope_degrees = _epitope_degrees(
            recognition_probability, clonotypes, epitopes
        )
    elif all([isinstance(current, int) for current in epitopes]):
        epitope_degrees = epitopes
    else:
        return -1

    value = 0
    for degree_0, degree_1 in combinations(epitope_degrees, 2):
        value += ((degree_0 * degree_1) ** 2 * (clonotypes - 1)) / (
            ((degree_0 + degree_1 - 2) * (clonotypes**4))
            - ((degree_0 * degree_1) ** 2 * (clonotypes - 1))
        )
    return value / comb(epitopes, 2)


def _preferential_attachment(
    recognition_probability, clonotypes, epitopes, crossreactivity, **kwargs
):
    """
    Calculates the clustering coefficient for a preferential attachment network.

    Parameters
    ----------
    recognition_probability : float
        Base epitope recognition probability.
    clonotypes : int
        Number of clonotypes in the network.
    epitopes : int
        Number of epitopes in the network.
    crossreactivity : float
        Base cross-reactivity probability.

    Returns
    -------
    float
        Clustering coefficient for a preferential attachment network.
    """
    clustering = 0
    degree = _preferential_degree(
        recognition_probability, clonotypes, epitopes, crossreactivity
    )
    probabilities, _ = _preferential_probabilities(
        recognition_probability, clonotypes, epitopes, crossreactivity
    )
    for current_clone in range(clonotypes):
        butterflies = 0
        for clone in [i for i in range(clonotypes) if i != current_clone]:
            butterflies += probabilities[clone] ** 2
        butterflies *= probabilities[current_clone] ** 2
        clustering += butterflies / ((2 * degree) - 2 - butterflies)
    return clustering / clonotypes


def _preferential_degree(
    recognition_probability, clonotypes, epitopes, crossreactivity
):
    """
    Calculates the expected value of the degree of VDP nodes in a preferential attachment network.

    Parameters
    ----------
    recognition_probability : float
        Base epitope recognition probability.
    clonotypes : int
        Number of clonotypes in the network.
    epitopes : int
        Number of epitopes in the network.
    crossreactivity : float
        Base cross-reactivity probability.

    Returns
    -------
    float
        Expected value of the degree of VDP nodes.
    """
    probabilities, crossreactivities = _preferential_probabilities(
        recognition_probability, clonotypes, epitopes, crossreactivity
    )
    value = 0
    for current_clone in range(clonotypes - 1):
        value += (
            (clonotypes - current_clone - 1)
            * probabilities[current_clone]
            * crossreactivities[current_clone]
        )
    return (recognition_probability * clonotypes) + (
        (1 - recognition_probability) * value
    )


def _preferential_probabilities(
    recognition_probability, clonotypes, epitopes, crossreactivity
):
    """
    Calculates the preferential attachment probabilities and the cross-reactivity probabilities for a preferential attachment network.

    Parameters
    ----------
    recognition_probability : float
        Base epitope recognition probability.
    clonotypes : int
        Number of clonotypes in the network.
    epitopes : int
        Number of epitopes in the network.
    crossreactivity : float
        Base cross-reactivity probability.

    Returns
    -------
    tuple[list[float]]
        List of preferential attachment probabilities and list of cross-reactivity probabilities
    """
    probabilities = [recognition_probability]
    for current_probability in range(1, clonotypes):
        current_value = 0
        for current_clone in range(current_probability):
            current_value += probabilities[
                current_clone
            ] * _preferential_cross_reactivity(
                probabilities, current_clone, epitopes, crossreactivity
            )
        probabilities.append(
            recognition_probability + ((1 - recognition_probability) * current_value)
        )
    crossreactivities = [
        _preferential_cross_reactivity(probabilities, clone, epitopes, crossreactivity)
        for clone in range(clonotypes)
    ]
    return probabilities, crossreactivities


def _preferential_cross_reactivity(probabilities, clonotype, epitopes, crossreactivity):
    """
    Calculates the cross-reactive recognition probability for ``clonotype`` in a network with ``epitopes`` number of epitopes, a base cross-reactivity probability of ``crossreactivity``.

    Parameters
    ----------
    probabilities : list[float]
        List of preferential attachment probabilities.
    clonotype : int
        Clonotype for which the probability is calculated.
    epitopes : int
        Number of epitopes in the network.
    crossreactivity : float
        Base cross-reactivity probability.

    Returns
    -------
    float
        Cross-reactive recognition probability.
    """
    value = 0
    for current_epitope in range(1, epitopes + 1):
        current_value = comb(epitopes, current_epitope)
        current_value *= (
            probabilities[0] * probabilities[clonotype]
        ) ** current_epitope
        current_value *= (1 - (probabilities[0] * probabilities[clonotype])) ** (
            epitopes - current_epitope
        )
        value += (
            current_value
            * (current_epitope / epitopes)
            * crossreactivity
            * probabilities[0]
        )
    return value
