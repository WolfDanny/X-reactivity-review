# %% Packages


from adaptive import *
from copy import deepcopy
from random import seed, getstate, setstate
import pickle

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

naive_homeostatic_value = 10.0  # \varphi_{i}
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
