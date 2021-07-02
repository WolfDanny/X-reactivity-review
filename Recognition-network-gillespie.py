# %% Packages


import math
import pickle
import numpy as np
from copy import deepcopy
from itertools import chain, combinations
from random import seed, uniform, sample, getstate, setstate

# %% Parameters


seed('Building a random recognition network')  # Seed to generate the networks
reproduce_last_result = True  # Reproduces the results from the previous run. If False or the seed.bin file is not found a new seed is used and saved to seed.bin

network = 0  # 0 = Unfocussed, 1 = Fixed degree, 2 = Preferential attachment
num_clonotypes = 3  # Number of clonotypes
starting_cells = 5  # Starting number of cells for every clonotype
num_peptides = 18  # Number of peptides

peptide_degree = 8  # Number of recognised peptides on the focussed degree network
peptide_probability = None  # Probability that a peptide will be recognised by a clonotype. If None the probability is calculated as peptide_degree / num_peptides

infection_duration = 1  # Duration of the infectious window [weeks]
first_infection_start = 1  # Time at which the first infection starts [months]
second_infection_start = 7  # Time at which the second infection starts [months]
total_duration = 12  # Total duration of the simulation [months]

naive_homeostatic_value = 10.0  # \varhpi_{i}
memory_homeostatic_value = 1.0  # \sigma_{M}
naive_matrix_value = np.asarray([[4/9, 2/9, 2/9, 1/9],  # p_{1}, p_{12}, p_{13}, p_{123}
                                 [4/9, 2/9, 2/9, 1/9],  # p_{2}, p_{21}, p_{23}, p_{213}
                                 [4/9, 2/9, 2/9, 1/9]])  # p_{3}, p_{31}, p_{32}, p_{312}

naive_death_value = 1.0  # \mu_{N}
memory_death_value = 0.8  # \mu_{M}
effector_death_value = 20.0  # \mu_{E}

naive_differentiation_constant_value = 1.0  # \alpha_{N}
memory_differentiation_constant_value = 2.0  # \alpha_{M}
effector_division_constant_value = 1.0  # \lambda_{E}

effector_differentiation_fraction = 0.1  # \beta
peptide_stimulus_value = 1000.0  # \gamma(v)

realisations = 1  # Number of realisations

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
        Returns the total number of cells that can receive stimulus from peptides, that is, it excludes effector cells that are already dividing.

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
                            if peptide_index not in self.peptides and peptide_index not in extra_peptides:
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
            self.memory -= 1
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


def gillespie_step(clone_list, time, division_time, current_infection, time_limit=None, peptide_list=None):
    """
    Performs a step of the Gillespie simulation and returns the updated state and time.

    If *time_limit* is included and the step makes *time* greater than *time_limit* returns the current state and *time_limit*.

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
        rate_list = np.append(rate_list, clone.birth_rate_effector(clone_list, peptide_list))
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
        rate_list = np.append(rate_list, clone.differentiation_rate_ne(clone_list, peptide_list))
    for clone in clone_list:
        rate_list = np.append(rate_list, clone.differentiation_rate_em(current_infection))
    for clone in clone_list:
        rate_list = np.append(rate_list, clone.differentiation_rate_me(clone_list, peptide_list))

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
        if (sum(rate_list[:rate]) / total_rate) <= r2 < (sum(rate_list[:rate + 1]) / total_rate):
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


if peptide_probability is None:
    peptide_prob_value = peptide_degree / num_peptides
else:
    peptide_prob_value = float(peptide_probability)

effector_division_time = 0.25 / 365  # 6 hour division time for effector cells during infection

effector_differentiation_value = (effector_differentiation_fraction / (1 - effector_differentiation_fraction)) * effector_death_value  # \psi_{E}
infection_time = infection_duration / 52
priming_start = first_infection_start / 12
priming_end = priming_start + infection_time
challenge_start = second_infection_start / 12
challenge_end = challenge_start + infection_time
experiment_end = total_duration / 12

# %% Gillespie algorithm


peptides = [Peptide(peptide_prob_value, i, peptide_stimulus_value) for i in range(num_peptides)]
initial_clones = []

for clone_index in range(num_clonotypes):
    initial_clones.append(Clonotype(clone_index, starting_cells, naive_homeostatic_value, naive_matrix_value, memory_homeostatic_value, effector_division_constant_value, naive_differentiation_constant_value, memory_differentiation_constant_value, naive_death_value, effector_death_value, memory_death_value, effector_differentiation_value))

    # Unfocussed network
    if network == 0:
        initial_clones[-1].add_peptide(peptides, 0)

    # Focussed network (degree)
    if network == 1:
        initial_clones[-1].add_peptide(peptides, 1, degree=peptide_degree)

    # Focussed network (preferential attachment)
    if network == 2:
        initial_clones[-1].add_peptide(peptides, 2, clone_list=initial_clones)

if reproduce_last_result:
    try:
        with open('seed.bin', 'rb') as file:
            seed_value = pickle.load(file)
            setstate(seed_value)
    except FileNotFoundError:
        print('Could not find seed.bin. A random realisation will be run instead and the seed will be saved to seed.bin.')
        reproduce_last_result = False
        seed()
else:
    seed()

seed_value = getstate()

states = []
naive = []
effector = []
memory = []
times = []

for current_realisation in range(realisations):

    current_clones = deepcopy(initial_clones)

    realisation_states = [deepcopy([clone.cells() for clone in current_clones])]
    realisation_naive = [deepcopy([clone.naive for clone in current_clones])]
    realisation_effector = [deepcopy([clone.effector + clone.effector_dividing for clone in current_clones])]
    realisation_memory = [deepcopy([clone.memory for clone in current_clones])]
    realisation_times = [0.0]

    current_time = 0.0

    # Gillespie algorithm

    # Pre-infection stage
    while current_time != priming_start:
        current_clones, current_time = gillespie_step(current_clones, current_time, effector_division_time, 0, priming_start)

        realisation_states.append(deepcopy([clone.cells() for clone in current_clones]))
        realisation_naive.append(deepcopy([clone.naive for clone in current_clones]))
        realisation_effector.append(deepcopy([clone.effector + clone.effector_dividing for clone in current_clones]))
        realisation_memory.append(deepcopy([clone.memory for clone in current_clones]))
        realisation_times.append(deepcopy(current_time))

        if [clone.cells() for clone in current_clones] == [0 for _ in current_clones]:
            break
    if [clone.cells() for clone in current_clones] == [0 for _ in current_clones]:
        states.append(deepcopy(realisation_states))
        naive.append(deepcopy(realisation_naive))
        effector.append(deepcopy(realisation_effector))
        memory.append(deepcopy(realisation_memory))
        times.append(deepcopy(realisation_times))
        continue

    # Priming stage
    while current_time != priming_end:
        current_clones, current_time = gillespie_step(current_clones, current_time, effector_division_time, peptide_stimulus_value, priming_end, peptides[:int(len(peptides) / 2)])

        realisation_states.append(deepcopy([clone.cells() for clone in current_clones]))
        realisation_naive.append(deepcopy([clone.naive for clone in current_clones]))
        realisation_effector.append(deepcopy([clone.effector + clone.effector_dividing for clone in current_clones]))
        realisation_memory.append(deepcopy([clone.memory for clone in current_clones]))
        realisation_times.append(deepcopy(current_time))

        if [clone.cells() for clone in current_clones] == [0 for _ in current_clones]:
            break
    if [clone.cells() for clone in current_clones] == [0 for _ in current_clones]:
        states.append(deepcopy(realisation_states))
        naive.append(deepcopy(realisation_naive))
        effector.append(deepcopy(realisation_effector))
        memory.append(deepcopy(realisation_memory))
        times.append(deepcopy(realisation_times))
        continue

    # Memory stage
    while current_time != challenge_start:
        current_clones, current_time = gillespie_step(current_clones, current_time, effector_division_time, 0, challenge_start)

        realisation_states.append(deepcopy([clone.cells() for clone in current_clones]))
        realisation_naive.append(deepcopy([clone.naive for clone in current_clones]))
        realisation_effector.append(deepcopy([clone.effector + clone.effector_dividing for clone in current_clones]))
        realisation_memory.append(deepcopy([clone.memory for clone in current_clones]))
        realisation_times.append(deepcopy(current_time))

        if [clone.cells() for clone in current_clones] == [0 for _ in current_clones]:
            break
    if [clone.cells() for clone in current_clones] == [0 for _ in current_clones]:
        states.append(deepcopy(realisation_states))
        naive.append(deepcopy(realisation_naive))
        effector.append(deepcopy(realisation_effector))
        memory.append(deepcopy(realisation_memory))
        times.append(deepcopy(realisation_times))
        continue

    # Challenge stage
    while current_time != challenge_end:
        current_clones, current_time = gillespie_step(current_clones, current_time, effector_division_time, peptide_stimulus_value, challenge_end, peptides[int(len(peptides) / 2):])

        realisation_states.append(deepcopy([clone.cells() for clone in current_clones]))
        realisation_naive.append(deepcopy([clone.naive for clone in current_clones]))
        realisation_effector.append(deepcopy([clone.effector + clone.effector_dividing for clone in current_clones]))
        realisation_memory.append(deepcopy([clone.memory for clone in current_clones]))
        realisation_times.append(deepcopy(current_time))

        if [clone.cells() for clone in current_clones] == [0 for _ in current_clones]:
            break
    if [clone.cells() for clone in current_clones] == [0 for _ in current_clones]:
        states.append(deepcopy(realisation_states))
        naive.append(deepcopy(realisation_naive))
        effector.append(deepcopy(realisation_effector))
        memory.append(deepcopy(realisation_memory))
        times.append(deepcopy(realisation_times))
        continue

    # Post-infection stage
    while current_time < experiment_end:
        current_clones, current_time = gillespie_step(current_clones, current_time, effector_division_time, 0)

        realisation_states.append(deepcopy([clone.cells() for clone in current_clones]))
        realisation_naive.append(deepcopy([clone.naive for clone in current_clones]))
        realisation_effector.append(deepcopy([clone.effector + clone.effector_dividing for clone in current_clones]))
        realisation_memory.append(deepcopy([clone.memory for clone in current_clones]))
        realisation_times.append(deepcopy(current_time))

        if [clone.cells() for clone in current_clones] == [0 for _ in current_clones]:
            break

    states.append(deepcopy(realisation_states))
    naive.append(deepcopy(realisation_naive))
    effector.append(deepcopy(realisation_effector))
    memory.append(deepcopy(realisation_memory))
    times.append(deepcopy(realisation_times))

# %% Storing results


with open('Data.bin', 'wb') as file:
    data = (states, times, priming_start, priming_end, challenge_start, challenge_end, experiment_end, network, initial_clones, peptides, naive, effector, memory)
    pickle.dump(data, file)

with open('Parameters.bin', 'wb') as file:
    data = ('num_clonotypes, num_peptides, starting_cells, naive_homeostatic_value, naive_matrix_value, memory_homeostatic_value, naive_death_value, memory_death_value, effector_death_value, effector_differentiation_value, memory_differentiation_constant_value, peptide_stimulus_value, peptide_degree, peptide_prob_value, realisations, effector_division_time, infection_time, experiment_end', num_clonotypes, num_peptides, starting_cells, naive_homeostatic_value, naive_matrix_value, memory_homeostatic_value, naive_death_value, memory_death_value, effector_death_value, effector_differentiation_value, memory_differentiation_constant_value, peptide_stimulus_value, peptide_degree, peptide_prob_value, realisations, effector_division_time, infection_time, experiment_end)
    pickle.dump(data, file)

if not reproduce_last_result:
    with open('seed.bin', 'wb') as file:
        pickle.dump(seed_value, file)
