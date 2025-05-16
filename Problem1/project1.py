N=10
M=30

Ln=[0]*N
Cn=[0]*N
Snm=[[0]*M for _ in range(N)]
Pn=[0]*N
qn=[0]*N
Tn=[0]*N
zn=[0]*N
yn=[0]*N
dn=[0]*N
En=[0]*N
Vm=[0]*M
Wm=[0]*M
tnm=[[0]*M for _ in range(N)]

# Objective Function
def average_task_completion_time(N, M, Ln, Cn, Snm, Pn, dn, En, zn, Vm, tnm):
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

