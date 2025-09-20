# file: main.py

import random
import matplotlib.pyplot as plt
from components import PM, VM
from algorithms import first_fit, best_fit, ACO_VMP

def calculate_metrics(pms):
    """Calculates performance metrics from a list of PMs after placement."""
    active_pms = [pm for pm in pms if pm.vms]
    if not active_pms:
        return 0, 0, 0
    
    total_energy = sum(p.get_power_consumption() for p in active_pms)
    sla_violations = sum(1 for p in active_pms if p.get_cpu_utilization() > 0.90)
    avg_utilization = sum(p.get_cpu_utilization() for p in active_pms) / len(active_pms)
    
    return total_energy, sla_violations, avg_utilization * 100

def run_simulation():
    """Main function to run the entire simulation and comparison."""
    # [cite_start]--- Simulation Environment Setup --- [cite: 91, 92]
    NUM_PMS = 50
    NUM_VMS = 200
    
    print("Setting up simulation environment...")
    pms = [PM(i, cpu_cap=100, mem_cap=128) for i in range(NUM_PMS)]
    vms = [VM(i, cpu_req=random.randint(5, 20), mem_req=random.randint(8, 32)) for i in range(NUM_VMS)]

    results = {}

    # [cite_start]--- Run Baseline Algorithms --- [cite: 94]
    print("\nRunning First Fit algorithm...")
    ff_pms = first_fit(vms, pms)
    results['First Fit'] = calculate_metrics(ff_pms)

    print("Running Best Fit algorithm...")
    bf_pms = best_fit(vms, pms)
    results['Best Fit'] = calculate_metrics(bf_pms)

    # --- Run MO-ACO Algorithm ---
    print("Running MO-ACO algorithm...")
    aco_solver = ACO_VMP(vms, pms)
    aco_pms = aco_solver.run()
    results['MO-ACO'] = calculate_metrics(aco_pms)
    
    # --- Print and Plot Results ---
    print("\n--- Simulation Results ---")
    print(f"{'Algorithm':<12} | {'Energy (W)':<12} | {'SLA Violations':<16} | {'Avg CPU Util (%)':<20}")
    print("-" * 70)
    for name, metrics in results.items():
        energy, sla, util = metrics
        print(f"{name:<12} | {energy:<12.2f} | {sla:<16} | {util:<20.2f}")

    # Plotting
    algorithms = list(results.keys())
    energy_data = [results[alg][0] for alg in algorithms]
    sla_data = [results[alg][1] for alg in algorithms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.bar(algorithms, energy_data, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Total Energy Consumption Comparison', fontsize=14)
    ax1.set_ylabel('Energy (Watts)', fontsize=12)
    ax1.tick_params(axis='x', rotation=15)
    
    ax2.bar(algorithms, sla_data, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('SLA Violation Comparison', fontsize=14)
    ax2.set_ylabel('Number of Overloaded PMs', fontsize=12)
    ax2.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run_simulation()