import random

#Sets and Indices 
N=10                                #𝒩={1,2,…,𝑁}  : Set of users/tasks 
M=30                                #ℳ={1,2,…,𝑀} : Set of local servers 

#Parameters 
#For each user 𝑛 ∈ 𝒩
Pn=[0]*N                            #: Number of CPU cycles required by task n
qn=[0]*N                            #: Memory required by task n
Tn=[0]*N                            #: Deadline for task n
zn=[0]*N                            #: User device processor speed (cycles per second) 
yn=[0]*N                            #: Memory available at the user n device
dn=[0]*N                            #: Transmission time from user n to the cloud
En=[0]*N                            #: Execution time of user n task on the cloud (fixed)

#For each server 𝑚 ∈ ℳ: 
Vm=[0]*M                            #: Processor speed of server m 
Wm=[0]*M                            #: Available memory at server m
tnm=[[0]*M for _ in range(N)]       #: Transmission time from user n to server m

#Decision Variables
Ln=[0]*N                            #: 1 if task n is executed locally zero other wise                  
Cn=[0]*N                            #: 1 if task n is executed to the cloud zero otherwise
Snm=[[0]*M for _ in range(N)]       #: 1 if task n is offloaded to server m and zero otherwise

# Objective Function
def average_task_completion_time(N, M, Ln, Cn, Snm, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm):
    total_time = 0

    for n in range(N):
        local_time = Ln[n] * (Pn[n] / zn[n])  # Local execution time
        cloud_time = Cn[n] * (2 * dn[n] + En[n])  # Cloud execution time

        server_time = 0
        for m in range(M):
            if Snm[n][m] == 1:
                # Sum of processing cycles of all tasks on server m
                total_processing_on_m = sum(Pn[i] * Snm[i][m] for i in range(N))
                server_time += Snm[n][m] * (2 * tnm[n][m] + (total_processing_on_m / Vm[m]))

        total_time += local_time + cloud_time + server_time

    return total_time / N

# Constraints
def verify_constraints(N, M, Ln, Cn, Snm, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm):
    # Constraint 1: Task Assignment Constraint
    for n in range(N):
        total_assignment = Ln[n] + Cn[n] + sum(Snm[n][m] for m in range(M))
        if total_assignment != 1:
            return False, f"Constraint 1 violated for task {n}: Total assignment = {total_assignment}"

    # Constraint 2: User Device Memory Constraint
    for n in range(N):
        if Ln[n] == 1 and qn[n] > yn[n]:
            return False, f"Constraint 2 violated for task {n}: qn > yn"

    # Constraint 3: Local Server Memory Constraint
    for m in range(M):
        total_memory_on_m = sum(qn[n] * Snm[n][m] for n in range(N))
        if total_memory_on_m > Wm[m]:
            return False, f"Constraint 3 violated for server {m}: memory used = {total_memory_on_m}, limit = {Wm[m]}"

    # Constraint 4: Deadline Constraint
    for n in range(N):
        local_time = Ln[n] * (Pn[n] / zn[n]) if zn[n] != 0 else 0
        cloud_time = Cn[n] * (2 * dn[n] + En[n])
        server_time = 0
        for m in range(M):
            if Snm[n][m] == 1:
                total_processing = sum(Pn[i] * Snm[i][m] for i in range(N))
                server_time += Snm[n][m] * (2 * tnm[n][m] + (total_processing / Vm[m] if Vm[m] != 0 else float('inf')))
        total_time = local_time + cloud_time + server_time
        if total_time > Tn[n]:
            return False, f"Constraint 4 violated for task {n}: total_time = {total_time}, deadline = {Tn[n]}"

    # Constraint 5: Binary Variables
    for n in range(N):
        if Ln[n] not in [0, 1]:
            return False, f"Constraint 5 violated: Ln[{n}] = {Ln[n]}"
        if Cn[n] not in [0, 1]:
            return False, f"Constraint 5 violated: Cn[{n}] = {Cn[n]}"
        for m in range(M):
            if Snm[n][m] not in [0, 1]:
                return False, f"Constraint 5 violated: Snm[{n}][{m}] = {Snm[n][m]}"

    return True, "All constraints satisfied"

# Encoding
def encode(N,M,Ln,Cn,Snm):
    encoded_tasks=[0]*N
    for n in range (N):
        if Ln[n]==1:
            encoded_tasks[n]=-1
            continue
        if Cn[n]==1:
            encoded_tasks[n]=-2
            continue
        for m in range(M):
            if Snm[n][m]==1:
                encoded_tasks[n]=m
                break
    return encoded_tasks

# Decoding
def decode_chromosome(chromosome, N, M):
    Ln = [0] * N
    Cn = [0] * N
    Snm = [[0] * M for _ in range(N)]

    for n in range(N):
        if chromosome[n] == -1:
            Ln[n] = 1
        elif chromosome[n] == -2:
            Cn[n] = 1
        elif 0 <= chromosome[n] < M:
            Snm[n][chromosome[n]] = 1
        else:
            raise ValueError(f"Invalid encoding value at index {n}: {chromosome[n]}")

    return Ln, Cn, Snm

# Generate the Population
def generate_population(N, M, population_size, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm):
    population = []
    
    while len(population) < population_size:
        chromosome = []
        for n in range(N):
            # Randomly assign a task: -1 (local), -2 (cloud), or 0 to M-1 (server)
            assignment = random.choice([-1, -2] + list(range(M)))
            chromosome.append(assignment)
        
        # Decode to decision variables
        Ln, Cn, Snm=decode_chromosome(chromosome, N, M)
        
        
        # Validate solution
        if verify_constraints(N, M, Ln, Cn, Snm, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm):
            population.append(chromosome)
    
    return population

# Selecting the elite Parents
def select_top_k_percent(population, K_percent, N, M, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm):
    scored_population = []

    for parent in population:
        Ln, Cn, Snm = decode_chromosome(parent, N, M)
        fitness = average_task_completion_time(N, M, Ln, Cn, Snm, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm)
        scored_population.append((parent, fitness))

    # Sort by fitness value (lower is better)
    scored_population.sort(key=lambda x: x[1])

    # Select top K%
    num_selected = max(1, int(len(population) * K_percent / 100))
    top_k = [parent for parent, _ in scored_population[:num_selected]]

    return top_k

# Crossover to generate 2 children
def crossover_chromosomes(parent1, parent2, N, M, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm):
    max_attempts = 10  # limit retries in case of invalid children
    for _ in range(max_attempts):
        # Choose a crossover point
        crossover_point = random.randint(1, N - 1)  # {1,2,...,N-1}

        # Create child by combining parts from parents
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        children=[]
        # Decode and check validity
        Ln, Cn, Snm = decode_chromosome(child1, N, M)
        if verify_constraints(N, M, Ln, Cn, Snm, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm):
            children.append(child1)

        Ln, Cn, Snm = decode_chromosome(child2, N, M)
        if verify_constraints(N, M, Ln, Cn, Snm, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm):
            children.append(child2)
        
        if len(children)==2:
            return children

    # If no valid child found, return one of the parents (fallback)
    return [parent1, parent2]

# Mutation
def mutate_chromosome(chromosome, N, M, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm):
    max_attempts = 10

    for _ in range(max_attempts):
        mutated = chromosome.copy()

        # Randomly select a task to mutate
        idx = random.randint(0, N - 1)

        # Select a new assignment different from current one
        options = [-1, -2] + list(range(M))
        options.remove(mutated[idx])  # remove current value to ensure change
        mutated[idx] = random.choice(options)

        # Decode and check validity
        Ln, Cn, Snm = decode_chromosome(mutated, N, M)
        if verify_constraints(N, M, Ln, Cn, Snm, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm):
            return mutated

    # If all attempts failed, return original
    return chromosome

# Genetic Algorithm Implementation
def genetic_algorithm(N: int, M: int,
    Pn: list[int], qn: list[int], Tn: list[int], zn: list[int], yn: list[int], dn: list[int], En: list[int],
    Vm: list[int], Wm: list[int], tnm: list[list[int]],
    max_iterations: int, population_size: int, K_percent: float, prop=0.05, early_stop=10, verbose=True):
    """
    Applies a Genetic Algorithm to minimize the average task completion time for N user tasks.

    This approach searches for the best assignment of each task to either:
    - be executed locally,
    - offloaded to the cloud, or
    - offloaded to one of the M available edge servers.

    Parameters:
    -----------
    N : int                     Number of tasks/users.
    M : int                     Number of local servers.
    Pn : list[int]              CPU cycles required for each task.
    qn : list[int]              Memory required for each task.
    Tn : list[int]              Deadline for each task.
    zn : list[int]              Processor speed of each user device.
    yn : list[int]              Available memory at each user device.
    dn : list[int]              Transmission time from user to cloud.
    En : list[int]              Execution time of the task on the cloud.
    Vm : list[int]              Processor speed of each local server.
    Wm : list[int]              Available memory at each server.
    tnm : list[list[int]]       Transmission time from user n to server m.
    max_iterations : int        Maximum number of iterations for the algorithm.
    population_size : int       Number of chromosomes (solutions) in each generation.
    K_percent : float           Top K% of the population to retain as elite parents.
    prop : float                Mutation probability (default = 0.05).
    early_stop : int            stops early if no improvement for specified iterations (default = 10).
    verbose : bool              If True, prints iteration details and the final solution (default = True).

    Returns:
    --------
    - best_overall_chromosome : list[int]
            The encoded best solution found.
    - best_overall_fitness : float
            The minimum average task completion time achieved by the best solution.

    Notes:
    ------
    - The population is evolved by selection by elitism (top k% fittest), random crossover, and random mutation.
    - Feasibility checks are applied within generation/mutation functions.
    - The algorithm stops early if the best solution does not improve for 10 consecutive generations.
    """

    best_overall_chromosome = None
    best_overall_fitness = float('inf')

    repeated_same_optimum = 0
    numer_of_iterations = 0

    while numer_of_iterations <= max_iterations:
        numer_of_iterations += 1
        if verbose:
            print(f"\n--- Iteration {numer_of_iterations} ---")

        # Generate initial population
        population = generate_population(N, M, population_size, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm)

        # Select elite group
        population_elite = select_top_k_percent(population, K_percent, N, M, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm)

        # Generate children via crossover
        new_population = []
        while len(new_population) + len(population_elite) < population_size:
            parent1 = random.choice(population_elite)
            parent2 = random.choice(population_elite)
            children = crossover_chromosomes(parent1, parent2, N, M, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm)
            new_population.extend(children)

        # Apply mutation
        mutated_population = []
        for chromosome in new_population:
            if random.random() < prop:
                chromosome = mutate_chromosome(chromosome, N, M, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm)
            mutated_population.append(chromosome)

        # Evaluate total population
        population = population_elite + mutated_population
        population_with_fitness = []
        for chromosome in population:
            Ln, Cn, Snm = decode_chromosome(chromosome, N, M)
            fitness = average_task_completion_time(N, M, Ln, Cn, Snm, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm)
            
            population_with_fitness.append((chromosome, fitness))

        # Sort by fitness
        population_with_fitness.sort(key=lambda x: x[1])
        best_chromosome, best_fitness = population_with_fitness[0]
        if verbose:
            print(f"Best fitness in this generation: {best_fitness:.4f}")

        # Check for improvement
        if best_fitness < best_overall_fitness:
            best_overall_chromosome = best_chromosome
            best_overall_fitness = best_fitness
            repeated_same_optimum = 0
            if verbose:
                print(">> New best solution found!")
        else:
            repeated_same_optimum += 1
            if verbose:
                print(f"No improvement. Counter: {repeated_same_optimum}/{early_stop}")

        # Stop if no improvement for {early_stop} iterations
        if repeated_same_optimum >= early_stop:
            if verbose:
                print("\n>> Stopping early: no improvement in {early_stop} consecutive generations.")
            break

    # Decode best result
    Ln, Cn, Snm = decode_chromosome(best_overall_chromosome, N, M)

    if verbose:
        print("\n=== Final Best Solution ===")
        print(f"Encoded Best Chromosome: {best_overall_chromosome}")
        print("\nDecoded Decision Variables:")
        print(f"Local Execution (Ln): {Ln}")
        print(f"Cloud Execution (Cn): {Cn}")
        print("Edge Server Assignment (Snm):")
        for row in Snm:
            print("   ",row)
        print(f"\nBest Average Task Completion Time: {best_overall_fitness:.4f}")

    return Ln, Cn, Snm, best_overall_fitness


