# file: main.py

import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from components import PM, VM
from algorithms import first_fit, best_fit, worst_fit, first_fit_decreasing, ACO_VMP

def calculate_metrics(pms, total_vms):
    """Calculates detailed performance metrics from a list of PMs after placement."""
    active_pms = [pm for pm in pms if pm.vms]
    placed_vms = sum(len(pm.vms) for pm in pms)
    unplaced_vms = max(0, total_vms - placed_vms)

    if not active_pms:
        return {
            'energy': 0.0,
            'sla_violations': 0,
            'avg_cpu_util_percent': 0.0,
            'cpu_util_std': 0.0,
            'avg_mem_util_percent': 0.0,
            'mem_util_std': 0.0,
            'active_pms': 0,
            'active_pm_ratio': 0.0,
            'placed_vms': placed_vms,
            'unplaced_vms': unplaced_vms,
            'unplaced_pct': (unplaced_vms / total_vms * 100.0) if total_vms else 0.0,
            'energy_per_vm': 0.0,
        }

    total_energy = sum(p.get_power_consumption() for p in active_pms)
    sla_violations = sum(1 for p in active_pms if p.get_cpu_utilization() > 0.90)

    cpu_utils = [p.get_cpu_utilization() for p in active_pms]
    mem_utils = [(p.mem_used / p.mem_cap) if p.mem_cap > 0 else 0.0 for p in active_pms]

    avg_cpu_util_percent = float(np.mean(cpu_utils) * 100.0) if cpu_utils else 0.0
    cpu_util_std = float(np.std(cpu_utils)) if cpu_utils else 0.0
    avg_mem_util_percent = float(np.mean(mem_utils) * 100.0) if mem_utils else 0.0
    mem_util_std = float(np.std(mem_utils)) if mem_utils else 0.0

    num_active_pms = len(active_pms)
    active_pm_ratio = num_active_pms / len(pms) if pms else 0.0

    energy_per_vm = (total_energy / placed_vms) if placed_vms > 0 else 0.0

    return {
        'energy': total_energy,
        'sla_violations': sla_violations,
        'avg_cpu_util_percent': avg_cpu_util_percent,
        'cpu_util_std': cpu_util_std,
        'avg_mem_util_percent': avg_mem_util_percent,
        'mem_util_std': mem_util_std,
        'active_pms': num_active_pms,
        'active_pm_ratio': active_pm_ratio,
        'placed_vms': placed_vms,
        'unplaced_vms': unplaced_vms,
        'unplaced_pct': (unplaced_vms / total_vms * 100.0) if total_vms else 0.0,
        'energy_per_vm': energy_per_vm,
    }

def run_simulation():
    """Main function to run the entire simulation and comparison."""
    # --- Simulation Environment Setup ---
    NUM_PMS = 50
    NUM_VMS = 200
    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("Setting up simulation environment...")
    pms = [PM(i, cpu_cap=100, mem_cap=128) for i in range(NUM_PMS)]
    vms = [VM(i, cpu_req=random.randint(5, 20), mem_req=random.randint(8, 32)) for i in range(NUM_VMS)]

    results = {}

    # --- Run Baseline Algorithms ---
    print("\nRunning First Fit algorithm...")
    ff_pms = first_fit(vms, pms)
    results['First Fit'] = calculate_metrics(ff_pms, NUM_VMS)

    print("Running Best Fit algorithm...")
    bf_pms = best_fit(vms, pms)
    results['Best Fit'] = calculate_metrics(bf_pms, NUM_VMS)
    
    print("Running Worst Fit algorithm...")
    wf_pms = worst_fit(vms, pms)
    results['Worst Fit'] = calculate_metrics(wf_pms, NUM_VMS)

    print("Running First Fit Decreasing algorithm...")
    ffd_pms = first_fit_decreasing(vms, pms)
    results['First Fit Decreasing'] = calculate_metrics(ffd_pms, NUM_VMS)

    # --- Run MO-ACO Algorithm ---
    print("\nRunning MO-ACO algorithm...")
    aco_solver = ACO_VMP(vms, pms, n_ants=25, n_iterations=80)
    aco_pms = aco_solver.run()
    results['MO-ACO'] = calculate_metrics(aco_pms, NUM_VMS)
    
    # --- Print and Plot Results ---
    print("\n--- Simulation Results ---")
    header = (
        f"{'Algorithm':<20} | {'Energy (W)':<12} | {'SLA':<5} | {'Avg CPU%':<9} | "
        f"{'Avg MEM%':<9} | {'Active':<6} | {'Unplaced':<9} | {'Energy/VM':<10}"
    )
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<20} | {m['energy']:<12.2f} | {m['sla_violations']:<5} | "
            f"{m['avg_cpu_util_percent']:<9.2f} | {m['avg_mem_util_percent']:<9.2f} | "
            f"{m['active_pms']:<6} | {m['unplaced_vms']:<9} | {m['energy_per_vm']:<10.2f}"
        )

    # Plotting
    algorithms = list(results.keys())
    energy_data = [results[alg]['energy'] for alg in algorithms]
    sla_data = [results[alg]['sla_violations'] for alg in algorithms]
    active_pms_data = [results[alg]['active_pms'] for alg in algorithms]
    unplaced_data = [results[alg]['unplaced_vms'] for alg in algorithms]
    energy_per_vm_data = [results[alg]['energy_per_vm'] for alg in algorithms]
    avg_cpu_pct_data = [results[alg]['avg_cpu_util_percent'] for alg in algorithms]
    avg_mem_pct_data = [results[alg]['avg_mem_util_percent'] for alg in algorithms]

    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#2ca02c']

    ax1.bar(algorithms, energy_data, color=colors)
    ax1.set_title('Total Energy (W)', fontsize=14)
    ax1.set_ylabel('Watts', fontsize=12)
    ax1.tick_params(axis='x', rotation=25)

    ax2.bar(algorithms, sla_data, color=colors)
    ax2.set_title('SLA Violations', fontsize=14)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.tick_params(axis='x', rotation=25)

    ax3.bar(algorithms, active_pms_data, color=colors)
    ax3.set_title('Active PMs', fontsize=14)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.tick_params(axis='x', rotation=25)

    bars4 = ax4.bar(algorithms, unplaced_data, color=colors)
    ax4.set_title('Unplaced VMs', fontsize=14)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.yaxis.set_major_locator(MaxNLocator(integer=True))
    try:
        ax4.bar_label(bars4, fmt='%.0f')
    except Exception:
        pass
    ax4.tick_params(axis='x', rotation=25)

    ax5.bar(algorithms, energy_per_vm_data, color=colors)
    ax5.set_title('Energy per Placed VM', fontsize=14)
    ax5.set_ylabel('Watts/VM', fontsize=12)
    ax5.tick_params(axis='x', rotation=25)

    ax6.bar(algorithms, avg_mem_pct_data, color=colors)
    ax6.set_title('Avg Memory Utilization (%)', fontsize=14)
    ax6.set_ylabel('%', fontsize=12)
    ax6.tick_params(axis='x', rotation=25)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_simulation()