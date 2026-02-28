from numpy.random import randint
from numpy.random import rand
import math
import random
from ikpy.chain import Chain
from ikpy.link import OriginLink
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from datetime import datetime

robot_chain = Chain.from_urdf_file("iiwa14.urdf", base_elements=["iiwa_link_0"])

def solve_ik(target):
    try:
        ik_solution = robot_chain.inverse_kinematics(target)
        return ik_solution[1:8]
    except Exception:
        return None

def get_jacobian(joint_values):
    thetas = joint_values[:6]
    c = np.cos
    s = np.sin
    c1, c2, c3, c4, c5, c6 = [c(t) for t in thetas]
    s1, s2, s3, s4, s5, s6 = [s(t) for t in thetas]
    d3, d5 = 0.42, 0.4  

    J51 = d3 * s4 * (c2 * c4 + c3 * s2 * s4) - d3 * c4 * (c2 * s4 - c3 * c4 * s2)
    J = np.array([
        [c2*s4 - c3*c4*s2, c4*s3, s4, 0, 0, -s5, c5*s6],
        [s2*s3, c3, 0, -1, 0, c5, s5*s6],
        [c2*s4 + c3*s2*s4, -s3*s4, c4, 0, 1, 0, c6],
        [d3*c4*s2*s3, d3*c3*c4, 0, 0, 0, -d5*c5, -d5*s5*s6],
        [J51, -d3*s3, 0, 0, 0, -d5*s5, d5*c5*s6],
        [-d3*s2*s3*s4, -d3*c3*s4, 0, 0, 0, 0, 0]
    ])
    return J

def compute_manipulability(joint_values):
    J = get_jacobian(joint_values)
    JJ_T = np.dot(J, J.T) 
    try:
        manipulability = np.sqrt(np.linalg.det(JJ_T))
    except np.linalg.LinAlgError:
        manipulability = 0.0
    return manipulability

JOINT_LIMITS = [
    (-2.967, 2.967),
    (-2.094, 2.094),
    (-2.967, 2.967),
    (-2.094, 2.094),
    (-2.967, 2.967),
    (-2.094, 2.094),
    (-3.054, 3.054),
]

def compute_joint_limit_distance(joint_values):
    if len(joint_values) > len(JOINT_LIMITS):
        joint_values = joint_values[1:]

    distances = []
    for idx, (val, (jmin, jmax)) in enumerate(zip(joint_values, JOINT_LIMITS)):
        dist_to_min = abs(val - jmin)
        dist_to_max = abs(jmax - val)
        joint_range = jmax - jmin
        norm_dist = min(dist_to_min, dist_to_max) / joint_range
        sinusoidal_component = np.sin(2 * np.pi * (val - jmin) / joint_range)
        modulated_norm_dist = norm_dist * (0.5 + 0.5 * sinusoidal_component)
        normalized_modulated_dist = (modulated_norm_dist - 0) / (1 - 0)
        normalized_modulated_dist = max(0, min(normalized_modulated_dist, 1))
        if normalized_modulated_dist < 1e-2:
            return 0.0
        distances.append(normalized_modulated_dist)
    return sum(distances) / len(distances)

def compute_singularity_metric(joint_values):
    J = get_jacobian(joint_values)
    cond_number = np.linalg.cond(J, 2)
    return 1.0/min(1000, cond_number)

def normalize_manipulability(value, max_value=0.175):
    return min(value / max_value, 1.0)


def objective(tcp_target):
    w2, w3, w4 = 1.0, 1.5, 2.0
    ik_solution = solve_ik(tcp_target)
    if ik_solution is None:
        return float('-inf')
    
    joint_limit_dist = compute_joint_limit_distance(ik_solution)
    singularity_metric = compute_singularity_metric(ik_solution)
    norm_manipulability = normalize_manipulability(compute_manipulability(ik_solution))
    
    
    score = (
        norm_manipulability * w2 +
        joint_limit_dist * w3 +
        singularity_metric * w4
    )
    return score

def decode(bounds, n_bits, bitstring):
    decoded = []
    largest = 2**n_bits
    for i in range(len(bounds)):
        start, end = i * n_bits, (i + 1) * n_bits
        substring = bitstring[start:end]
        chars = ''.join([str(s) for s in substring])
        integer = int(chars, 2)
        value = bounds[i][0] + (integer / largest) * (bounds[i][1] - bounds[i][0])
        decoded.append(value)
    return decoded

def selection(population, scores, k=3):
    selection_ix = randint(len(population))
    for ix in randint(0, len(population), k-1):
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    if rand() < r_cross:
        pt = randint(1, len(p1)-1)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

def mutation(bitstring, r_mut):
    rand_func = random.random
    for i in range(len(bitstring)):
        if rand_func() < r_mut:
            bitstring[i] = 1 - bitstring[i]
    return bitstring

def genetic_algorithm(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
    population = [randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(n_pop)]
    best, best_eval = 0, objective(decode(bounds, n_bits, population[0]))
    
    for gen in range(n_iter):
        decoded = [decode(bounds, n_bits, p) for p in population]
        scores = []
        
        for d in decoded:
            ik_solution = solve_ik(d)
            if ik_solution is None:
                score = float('-inf')
            else:
                raw_m = compute_manipulability(ik_solution)
                raw_jld = compute_joint_limit_distance(ik_solution)
                raw_s = compute_singularity_metric(ik_solution)
                
                m = normalize_manipulability(raw_m)
                jld = raw_jld
                s = raw_s
                
                score = m * 1.0 + jld * 1.5 + s * 2.0
            scores.append(score)
        
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = population[i], scores[i]
        
        selected = [selection(population, scores) for _ in range(n_pop)]
        children = []
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i + 1]
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
        population = children
    
    # Get final population scores for statistics
    final_decoded = [decode(bounds, n_bits, p) for p in population]
    final_scores = [objective(d) for d in final_decoded]
    # Filter out -inf values
    final_scores_valid = [s for s in final_scores if np.isfinite(s)]
    
    best_decoded = decode(bounds, n_bits, best)
    return best_decoded, best_eval, final_scores_valid

# Main execution
bounds = [
    (0, 0.6),
    (-0.45, 0.45),
    (0, 1.6),
]
n_bits = 16
n_iter = 100
n_pop = 100
r_cross = 0.9
r_mut = 1.0 / (n_bits * len(bounds))

# Run 30 times
n_runs = 30
results = []

print(f"Starting {n_runs} runs of the genetic algorithm...")
print("=" * 60)

for run in range(n_runs):
    start_time = time.time()
    
    # Run the genetic algorithm
    best_solution, best_score, final_pop_scores = genetic_algorithm(
        objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate statistics
    if len(final_pop_scores) > 0:
        std_dev = np.std(final_pop_scores)
        median = np.median(final_pop_scores)
        q1 = np.percentile(final_pop_scores, 25)
        q3 = np.percentile(final_pop_scores, 75)
        iqr = q3 - q1
    else:
        std_dev = 0
        median = 0
        q1 = 0
        q3 = 0
        iqr = 0
    
    # Store results
    results.append({
        'Run': run + 1,
        'Best_Solution_X': best_solution[0],
        'Best_Solution_Y': best_solution[1],
        'Best_Solution_Z': best_solution[2],
        'Best_Score': best_score,
        'Std_Dev': std_dev,
        'Median': median,
        'Q1': q1,
        'Q3': q3,
        'IQR': iqr,
        'Time_Seconds': elapsed_time
    })
    
    print(f"Run {run + 1}/{n_runs} completed in {elapsed_time:.2f}s - Best Score: {best_score:.4f}")

print("=" * 60)
print("All runs completed!")

# Create DataFrame
df = pd.DataFrame(results)

# Save to CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f'ga_results_{timestamp}.csv'
df.to_csv(csv_filename, index=False)
print(f"\nResults saved to: {csv_filename}")

# Save detailed summary
summary_filename = f'ga_summary_{timestamp}.txt'
with open(summary_filename, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("GENETIC ALGORITHM - 30 RUNS SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Number of runs: {n_runs}\n")
    f.write(f"Iterations per run: {n_iter}\n")
    f.write(f"Population size: {n_pop}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("OVERALL STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Best Score Across All Runs: {df['Best_Score'].max():.6f}\n")
    f.write(f"Worst Score Across All Runs: {df['Best_Score'].min():.6f}\n")
    f.write(f"Mean Best Score: {df['Best_Score'].mean():.6f}\n")
    f.write(f"Std Dev of Best Scores: {df['Best_Score'].std():.6f}\n")
    f.write(f"Median Best Score: {df['Best_Score'].median():.6f}\n\n")
    
    f.write(f"Average Time per Run: {df['Time_Seconds'].mean():.2f} seconds\n")
    f.write(f"Total Time: {df['Time_Seconds'].sum():.2f} seconds\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("DETAILED RESULTS BY RUN\n")
    f.write("=" * 80 + "\n\n")
    
    for idx, row in df.iterrows():
        f.write(f"Run {row['Run']}:\n")
        f.write(f"  Best Solution: ({row['Best_Solution_X']:.6f}, "
                f"{row['Best_Solution_Y']:.6f}, {row['Best_Solution_Z']:.6f})\n")
        f.write(f"  Best Score: {row['Best_Score']:.6f}\n")
        f.write(f"  Standard Deviation: {row['Std_Dev']:.6f}\n")
        f.write(f"  Median: {row['Median']:.6f}\n")
        f.write(f"  IQR (Q1-Q3): {row['Q1']:.6f} - {row['Q3']:.6f} (Range: {row['IQR']:.6f})\n")
        f.write(f"  Time: {row['Time_Seconds']:.2f} seconds\n\n")

print(f"Summary saved to: {summary_filename}")

# Display summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(df.to_string(index=False))

print("\n" + "=" * 80)
print("EXECUTION COMPLETE")
print("=" * 80)
