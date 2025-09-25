# file: algorithms.py

import numpy as np
import random
from components import PM

# --- Baseline Algorithms ---

def first_fit(vms, pms):
    """Places each VM on the first PM that can host it."""
    for pm in pms:
        pm.reset()
    for vm in vms:
        for pm in pms:
            if pm.place_vm(vm):
                break
    return pms

def best_fit(vms, pms):
    """Places each VM on the PM that provides the tightest fit (least remaining CPU)."""
    for pm in pms:
        pm.reset()
    for vm in vms:
        best_pm = None
        min_remaining_cpu = float('inf')
        for pm in pms:
            if pm.can_host(vm):
                remaining_cpu = pm.cpu_cap - (pm.cpu_used + vm.cpu)
                if remaining_cpu < min_remaining_cpu:
                    min_remaining_cpu = remaining_cpu
                    best_pm = pm
        if best_pm:
            best_pm.place_vm(vm)
    return pms

def worst_fit(vms, pms):
    """Places each VM on the PM that has the most remaining CPU capacity."""
    for pm in pms:
        pm.reset()
    for vm in vms:
        best_pm = None
        max_remaining_cpu = -1
        for pm in pms:
            if pm.can_host(vm):
                remaining_cpu = pm.cpu_cap - (pm.cpu_used + vm.cpu)
                if remaining_cpu > max_remaining_cpu:
                    max_remaining_cpu = remaining_cpu
                    best_pm = pm
        if best_pm:
            best_pm.place_vm(vm)
    return pms

def first_fit_decreasing(vms, pms):
    """Sorts VMs by CPU requirement in descending order, then applies First Fit."""
    for pm in pms:
        pm.reset()
    # Sort VMs in descending order of CPU requirement
    sorted_vms = sorted(vms, key=lambda vm: vm.cpu, reverse=True)
    for vm in sorted_vms:
        for pm in pms:
            if pm.place_vm(vm):
                break
    return pms

# --- Multi-Objective Ant Colony Optimization (MO-ACO) ---

class ACO_VMP:
    """
    Implements the MO-ACO algorithm based on the project document.
    It uses pheromones and heuristics to find an optimal VM placement.
    """
    def __init__(self, vms, pms, n_ants=10, n_iterations=50, alpha=1.0, beta=2.0, rho=0.1, w1=0.5, w2=0.3, w3=0.2):
        self.vms = vms
        self.pms = pms
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Pheromone influence
        self.beta = beta    # Heuristic influence
        self.rho = rho      # Pheromone evaporation rate
        # Weights for the fitness function
        self.w1, self.w2, self.w3 = w1, w2, w3 # Energy, Utilization, SLA Violations
        self.pheromone = np.ones((len(vms), len(pms)))

    def _heuristic_function(self, vm, pm):
        """Calculates heuristic value based on the formula in the document."""
        if not pm.can_host(vm): return 0
        
        # Invert utilization to get remaining capacity fraction
        cpu_rem = 1 - ((pm.cpu_used + vm.cpu) / pm.cpu_cap)
        mem_rem = 1 - ((pm.mem_used + vm.mem) / pm.mem_cap)
        
        # Energy factor: favors PMs that are already on but not heavily loaded
        energy_j = pm.get_power_consumption() + 1
        
        return (cpu_rem) * (mem_rem) * (1 / energy_j)

    def _calculate_fitness(self, pms_state):
        """Evaluates a solution using the multi-objective fitness function."""
        active_pms = [pm for pm in pms_state if pm.vms]
        if not active_pms: return float('inf')

        total_energy = sum(pm.get_power_consumption() for pm in active_pms)
        avg_util = np.mean([pm.get_cpu_utilization() for pm in active_pms])
        sla_violations = sum(1 for pm in active_pms if pm.get_cpu_utilization() > 0.90)

        # Fitness formula combines weighted objectives for minimization
        fitness = (self.w1 * total_energy +
                   self.w2 * (1 - avg_util) +  # We want to maximize utilization, so minimize (1 - util)
                   self.w3 * sla_violations)
        return fitness

    def run(self):
        """Executes the main ACO algorithm loop."""
        best_solution_mapping = None
        best_fitness = float('inf')

        for i in range(self.n_iterations):
            all_ant_solutions = []
            for _ in range(self.n_ants):
                temp_pms = [PM(p.id, p.cpu_cap, p.mem_cap) for p in self.pms]
                solution_mapping = {}
                
                vms_to_place = random.sample(self.vms, len(self.vms))
                for vm in vms_to_place:
                    pheromones = self.pheromone[vm.id]
                    heuristics = np.array([self._heuristic_function(vm, pm) for pm in temp_pms])
                    
                    probabilities = (pheromones ** self.alpha) * (heuristics ** self.beta)
                    
                    if np.sum(probabilities) == 0:
                        available_pms_idx = [idx for idx, p in enumerate(temp_pms) if p.can_host(vm)]
                        if not available_pms_idx: continue
                        chosen_pm_idx = random.choice(available_pms_idx)
                    else:
                        probabilities /= np.sum(probabilities)
                        chosen_pm_idx = np.random.choice(len(self.pms), p=probabilities)

                    if temp_pms[chosen_pm_idx].can_host(vm):
                        temp_pms[chosen_pm_idx].place_vm(vm)
                        solution_mapping[vm.id] = temp_pms[chosen_pm_idx].id

                fitness = self._calculate_fitness(temp_pms)
                all_ant_solutions.append((solution_mapping, fitness))

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution_mapping = solution_mapping

            # Update pheromones
            self.pheromone *= (1 - self.rho) # Evaporation
            if all_ant_solutions:
                best_iter_solution, best_iter_fitness = min(all_ant_solutions, key=lambda x: x[1])
                for vm_id, pm_id in best_iter_solution.items():
                    self.pheromone[vm_id, pm_id] += 1.0 / (best_iter_fitness + 1e-5) # Deposition
            
            print(f"Iteration {i+1}/{self.n_iterations}: Best Fitness = {best_fitness:.2f}")

        # Apply the best found mapping to the actual PMs
        for pm in self.pms: pm.reset()
        if best_solution_mapping:
            for vm_id, pm_id in best_solution_mapping.items():
                self.pms[pm_id].place_vm(self.vms[vm_id])
        return self.pms