# file: main.py

import random
import matplotlib.pyplot as plt
import numpy as np
from components import PM, VM
from algorithms import first_fit, best_fit, worst_fit, first_fit_decreasing, ACO_VMP

def calculate_metrics(pms):
    """Calculates performance metrics from a list of PMs after placement."""
    active_pms = [pm for pm in pms if pm.vms]
    if not active_pms:
        return 0, 0, 0, 0, 0
    
    total_energy = sum(p.get_power_consumption() for p in active_pms)
    sla_violations = sum(1 for p in active_pms if p.get_cpu_utilization() > 0.90)
    
    utilizations = [p.get_cpu_utilization() for p in active_pms]
    avg_utilization = np.mean(utilizations) * 100
    util_std_dev = np.std(utilizations)
    
    num_active_pms = len(active_pms)
    
    return total_energy, sla_violations, avg_utilization, num_active_pms, util_std_dev

def run_simulation():
    """Main function to run the entire simulation and comparison."""
    # --- Simulation Environment Setup ---
    NUM_PMS = 50
    NUM_VMS = 200
    
    print("Setting up simulation environment...")
    pms = [PM(i, cpu_cap=100, mem_cap=128) for i in range(NUM_PMS)]
    vms = [VM(i, cpu_req=random.randint(5, 20), mem_req=random.randint(8, 32)) for i in range(NUM_VMS)]

    results = {}

    # --- Run Baseline Algorithms ---
    print("\nRunning First Fit algorithm...")
    ff_pms = first_fit(vms, pms)
    results['First Fit'] = calculate_metrics(ff_pms)

    print("Running Best Fit algorithm...")
    bf_pms = best_fit(vms, pms)
    results['Best Fit'] = calculate_metrics(bf_pms)
    
    print("Running Worst Fit algorithm...")
    wf_pms = worst_fit(vms, pms)
    results['Worst Fit'] = calculate_metrics(wf_pms)

    print("Running First Fit Decreasing algorithm...")
    ffd_pms = first_fit_decreasing(vms, pms)
    results['First Fit Dec.'] = calculate_metrics(ffd_pms)

    # --- Run MO-ACO Algorithm ---
    print("\nRunning MO-ACO algorithm...")
    aco_solver = ACO_VMP(vms, pms)
    aco_pms = aco_solver.run()
    results['MO-ACO'] = calculate_metrics(aco_pms)
    
    # --- Print and Plot Results ---
    print("\n--- Simulation Results ---")
    header = f"{'Algorithm':<18} | {'Energy (W)':<12} | {'SLA Violations':<16} | {'Avg CPU Util (%)':<20} | {'Active PMs':<12} | {'CPU Util Std Dev':<20}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        energy, sla, util, active_pms, util_std = metrics
        print(f"{name:<18} | {energy:<12.2f} | {sla:<16} | {util:<20.2f} | {active_pms:<12} | {util_std:<20.4f}")

    # Plotting
    algorithms = list(results.keys())
    energy_data = [results[alg][0] for alg in algorithms]
    sla_data = [results[alg][1] for alg in algorithms]
    active_pms_data = [results[alg][3] for alg in algorithms]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#2ca02c']

    ax1.bar(algorithms, energy_data, color=colors)
    ax1.set_title('Total Energy Consumption Comparison', fontsize=14)
    ax1.set_ylabel('Energy (Watts)', fontsize=12)
    ax1.tick_params(axis='x', rotation=25)
    
    ax2.bar(algorithms, sla_data, color=colors)
    ax2.set_title('SLA Violation Comparison', fontsize=14)
    ax2.set_ylabel('Number of Overloaded PMs', fontsize=12)
    ax2.tick_params(axis='x', rotation=25)
    
    ax3.bar(algorithms, active_pms_data, color=colors)
    ax3.set_title('Number of Active PMs Comparison', fontsize=14)
    ax3.set_ylabel('Count of Active PMs', fontsize=12)
    ax3.tick_params(axis='x', rotation=25)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_simulation()