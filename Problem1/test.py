from Task_Offloading_Optimization import *
import random

def test_constraints1():
    # Example parameters
    N = 4               
    M = 5
    Pn = [100, 200, 150, 300, 250, 400, 350, 450, 500, 600]
    qn = [10, 20, 15, 30, 25, 40, 35, 45, 50, 60]
    Tn = [1000] * N
    zn = [1000] * N
    yn = [50] * N
    dn = [1] * N
    En = [5] * N
    Vm = [2000] * M
    Wm = [100] * M
    tnm = [[1 for _ in range(M)] for _ in range(N)]

    Ln = [0] * N
    Cn = [0] * N
    Snm = [[0] * M for _ in range(N)]

    # Example assignment
    Ln[0], Cn[1], Snm[2][3] = 1, 1, 1           #!!!Violation here 3 tasks were assigned instead of 4
    
    valid, message = verify_constraints(N, M, Ln, Cn, Snm, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm)
    
    print("Testing Constraints")
    print(f"Ln: {Ln} \nCn: {Cn}\nSnm:")     # For debugging purposes
    for row in Snm:
        print("   ",row)
    print("Constraints Valid!" if valid else f"Constraints Invalid: {message}")
    print("======================")

def test_constraints2():
    # Example parameters
    N = 3
    M = 5
    Pn = [100, 200, 150, 300, 250, 400, 350, 450, 500, 600]
    qn = [10, 20, 15, 30, 25, 40, 35, 45, 50, 60]
    Tn = [1000] * N
    zn = [1000] * N
    yn = [5] * N                    #!!!Violation here qn > yn 
    dn = [1] * N
    En = [5] * N
    Vm = [2000] * M
    Wm = [100] * M
    tnm = [[1 for _ in range(M)] for _ in range(N)]

    Ln = [0] * N
    Cn = [0] * N
    Snm = [[0] * M for _ in range(N)]

    # Example assignment
    Ln[0], Cn[1], Snm[2][3] = 1, 1, 1

    valid, message = verify_constraints(N, M, Ln, Cn, Snm, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm)
    
    print("Testing Constraints")
    print(f"Ln: {Ln} \nCn: {Cn}\nSnm:")     # For debugging purposes
    for row in Snm:
        print("   ",row)
    print("Constraints Valid!" if valid else f"Constraints Invalid: {message}")
    print("======================")

# Test for objective function
def test_objective_function():
    # Example parameters
    N = 3
    M = 5
    Pn = [100, 200, 150, 300, 250, 400, 350, 450, 500, 600]
    qn = [10, 20, 15, 30, 25, 40, 35, 45, 50, 60]
    Tn = [1000] * N
    zn = [1000] * N
    yn = [50] * N
    dn = [1] * N
    En = [5] * N
    Vm = [2000] * M
    Wm = [100] * M
    tnm = [[1 for _ in range(M)] for _ in range(N)]

    Ln = [0] * N
    Cn = [0] * N
    Snm = [[0] * M for _ in range(N)]

    # Example assignment
    Ln[0], Cn[1], Snm[2][3] = 1, 1, 1

    result = average_task_completion_time(N, M, Ln, Cn, Snm, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm)
    Expected = 3.0583
    print("Testing Objective Function (Expected: 3.0583)")
    print(f"Average Task Completion Time: {result}")
    print("Test Passed!" if abs(result - Expected) < 1e-4 else "Test Failed!")
    print ("======================")

# Test for Encoding
def test_encoding():
    # Example parameters
    N = 3
    M = 5
    Pn = [100, 200, 150, 300, 250, 400, 350, 450, 500, 600]
    qn = [10, 20, 15, 30, 25, 40, 35, 45, 50, 60]
    Tn = [1000] * N
    zn = [1000] * N
    yn = [50] * N
    dn = [1] * N
    En = [5] * N
    Vm = [2000] * M
    Wm = [100] * M
    tnm = [[1 for _ in range(M)] for _ in range(N)]

    Ln = [0] * N
    Cn = [0] * N
    Snm = [[0] * M for _ in range(N)]

    # Example assignment
    Ln[0], Cn[1], Snm[2][3] = 1, 1, 1
    encoding_result = encode(N, M, Ln, Cn, Snm)
    
    print("Testing Encoding")
    print(f"Ln: {Ln} \nCn: {Cn}\nSnm:")     # For debugging purposes
    for row in Snm:
        print("   ",row)
    print(f"Encoding Result: {encoding_result}")
    print("======================")

def test_encoding_plus_simple_cross():
    # Example parameters
    N = 6
    M = 4
    Ln = [1, 0, 0, 1, 0, 0]
    Cn = [0, 1, 0, 0, 1, 0]
    Snm = [
        [0, 0, 0, 0],  # Task 0 → local
        [0, 0, 0, 0],  # Task 1 → cloud
        [0, 1, 0, 0],  # Task 2 → server 1
        [0, 0, 0, 0],  # Task 3 → local
        [0, 0, 0, 0],  # Task 4 → cloud
        [0, 0, 1, 0],  # Task 5 → server 2
    ]
    encoding_result1 = encode(N, M, Ln, Cn, Snm)
    Ln = [1, 0, 0, 1, 0, 0]            # Task 0, 3 is local
    Cn = [0, 1, 0, 0, 1, 1]            # Task 1, 4, 5 goes to cloud
    Snm = [                            # Task 2 assigned to server 3
        [0, 0, 0, 0],  # n=0
        [0, 0, 0, 0],  # n=1
        [0, 0, 0, 1],  # n=2
        [0, 0, 0, 0],  # n=3
        [0, 0, 0, 0],  # n=4
        [0, 0, 0, 0],  # n=5
    ]

    encoding_result2 = encode(N, M, Ln, Cn, Snm)
    
    print("Testing Encoding + simple Crossover")

    print(f"Encoding Result1: {encoding_result1}")
    print(f"Encoding Result2: {encoding_result2}")
    crossover_point = 5
    child1 = encoding_result1[:crossover_point] + encoding_result2[crossover_point:]
    child2 = encoding_result2[:crossover_point] + encoding_result1[crossover_point:]
    print(f"Encoding child1: {child1}")
    print(f"Encoding child2: {child2}")

def main():
    print("Running Tests...")
    test_objective_function()
    test_constraints1()  
    test_constraints2()
    test_encoding()
    test_encoding_plus_simple_cross()

    # Sets and Indices
    N = 10  # Number of users/tasks
    M = 30  # Number of edge servers

    # Parameters for each user n ∈ 𝒩
    Pn = [random.randint(100, 1000) for _ in range(N)]     # CPU cycles in millions
    qn = [random.randint(50, 200) for _ in range(N)]       # Memory in MB
    Tn = [random.randint(1, 10) for _ in range(N)]         # Deadline in seconds
    zn = [random.randint(500, 2000) for _ in range(N)]     # User device processor speed in MHz
    yn = [random.randint(100, 512) for _ in range(N)]      # User device memory in MB
    dn = [random.uniform(0.1, 1.0) for _ in range(N)]      # Transmission time to cloud in seconds
    En = [random.uniform(0.5, 3.0) for _ in range(N)]      # Cloud execution time in seconds

    # Parameters for each server m ∈ ℳ
    Vm = [random.randint(1000, 5000) for _ in range(M)]    # Server processor speed in MHz
    Wm = [random.randint(512, 2048) for _ in range(M)]     # Server memory in MB

    # Transmission time from user n to server m
    tnm = [[random.uniform(0.05, 0.5) for _ in range(M)] for _ in range(N)]
    genetic_algorithm(N, M, Pn, qn, Tn, zn, yn, dn, En, Vm, Wm, tnm, max_iterations=500, population_size=50, K_percent=0.2)
if __name__ == "__main__":
    main()
