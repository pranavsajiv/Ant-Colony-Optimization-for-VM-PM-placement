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
    Implements a refined MO-ACO for VM placement with:
    - normalized multi-objective fitness (energy, SLA, utilization, unplaced VMs)
    - feasible-only sampling with candidate list (top-K by heuristic)
    - marginal-energy heuristic with new-PM penalty
    - Ant Colony System (ACS) style pheromone updates (local + global) with bounds
    - exploitation probability q0 and early-stopping
    """
    def __init__(
        self,
        vms,
        pms,
        n_ants=20,
        n_iterations=60,
        alpha=1.0,
        beta=3.0,
        rho=0.2,
        q0=0.9,
        xi=0.1,
        candidate_k=10,
        w1=0.5,
        w2=0.3,
        w3=0.1,
        w_unplaced=0.1,
        tau0=None,
        tau_min=1e-4,
        tau_max=None,
        patience=15
    ):
        self.vms = vms
        self.pms = pms
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Pheromone influence
        self.beta = beta    # Heuristic influence
        self.rho = rho      # Pheromone evaporation rate (global)
        self.q0 = q0        # Greedy choice probability (ACS)
        self.xi = xi        # Local evaporation factor (ACS)
        self.candidate_k = candidate_k
        # Weights for the fitness function (w1, w2, w3 kept for back-compat names)
        self.w1 = w1  # Energy
        self.w2 = w2  # SLA violations
        self.w3 = w3  # (1 - avg CPU utilization)
        self.w_unplaced = w_unplaced  # Unplaced VMs penalty

        # Pheromone initialization and bounds
        self.tau0 = (1.0 / max(1, len(pms))) if tau0 is None else tau0
        self.pheromone = np.full((len(vms), len(pms)), self.tau0, dtype=float)
        self.tau_min = tau_min
        self.tau_max = (10.0 * self.tau0) if tau_max is None else tau_max

        # Early stopping
        self.patience = patience

    def _heuristic_function(self, vm, pm, new_pm_penalty=0.5, eps=1e-9):
        """Marginal-energy heuristic with new-PM activation penalty and capacity balance."""
        if not pm.can_host(vm):
            return 0.0

        # Remaining capacity fractions if we place vm
        cpu_rem = 1.0 - ((pm.cpu_used + vm.cpu) / pm.cpu_cap)
        mem_rem = 1.0 - ((pm.mem_used + vm.mem) / pm.mem_cap)
        cpu_rem = max(cpu_rem, 0.0)
        mem_rem = max(mem_rem, 0.0)

        # Marginal energy of placing this VM
        def power_at_util(util):
            return pm.power_idle + (pm.power_max - pm.power_idle) * util

        util_now = pm.get_cpu_utilization()
        util_after = (pm.cpu_used + vm.cpu) / pm.cpu_cap
        marginal_energy = max(power_at_util(util_after) - power_at_util(util_now), 0.0)

        # Discourage activating a new PM
        activation_factor = (1.0 if pm.vms else new_pm_penalty)

        # Prefer PMs with better headroom and lower marginal energy increase
        return activation_factor * (cpu_rem * mem_rem) / (marginal_energy + eps)

    def _calculate_fitness(self, pms_state):
        """Normalized multi-objective fitness with unplaced penalty (minimize)."""
        active_pms = [pm for pm in pms_state if pm.vms]
        placed_vms = sum(len(pm.vms) for pm in pms_state)
        total_vms = len(self.vms)
        unplaced = max(0, total_vms - placed_vms)

        if not active_pms:
            return float('inf')

        total_energy = sum(pm.get_power_consumption() for pm in active_pms)
        avg_util = np.mean([pm.get_cpu_utilization() for pm in active_pms]) if active_pms else 0.0
        sla_violations = sum(1 for pm in active_pms if pm.get_cpu_utilization() > 0.90)

        # Normalization bases
        energy_max = sum(pm.power_max for pm in self.pms)  # upper bound if all PMs at 100%
        sla_max = max(1, len(self.pms))

        E = total_energy / (energy_max + 1e-9)
        S = sla_violations / sla_max
        U = (1.0 - avg_util)  # already in [0,1]
        P = unplaced / max(1, total_vms)

        return self.w1 * E + self.w2 * S + self.w3 * U + self.w_unplaced * P

    def _acs_local_update(self, vm_id, pm_id):
        # tau = (1 - xi) * tau + xi * tau0
        self.pheromone[vm_id, pm_id] = (1.0 - self.xi) * self.pheromone[vm_id, pm_id] + self.xi * self.tau0
        # enforce bounds
        if self.pheromone[vm_id, pm_id] < self.tau_min:
            self.pheromone[vm_id, pm_id] = self.tau_min
        elif self.pheromone[vm_id, pm_id] > self.tau_max:
            self.pheromone[vm_id, pm_id] = self.tau_max

    def run(self):
        """Executes the main ACO algorithm loop (ACS)."""
        best_solution_mapping = None
        best_fitness = float('inf')
        no_improve = 0

        for i in range(self.n_iterations):
            all_ant_solutions = []

            for _ in range(self.n_ants):
                # fresh PM copies for this ant
                temp_pms = [PM(p.id, p.cpu_cap, p.mem_cap) for p in self.pms]
                solution_mapping = {}

                # random VM order
                vms_to_place = random.sample(self.vms, len(self.vms))

                for vm in vms_to_place:
                    # Feasible PMs only
                    feasible_idxs = [idx for idx, p in enumerate(temp_pms) if p.can_host(vm)]
                    if not feasible_idxs:
                        continue

                    # Candidate list: top-K by heuristic
                    heuristics_full = np.array([self._heuristic_function(vm, temp_pms[j]) for j in feasible_idxs], dtype=float)
                    if heuristics_full.size == 0:
                        continue

                    # filter top-K
                    if self.candidate_k is not None and len(feasible_idxs) > self.candidate_k:
                        top_k_idx = np.argpartition(-heuristics_full, self.candidate_k - 1)[:self.candidate_k]
                        candidate_idxs = [feasible_idxs[k] for k in top_k_idx]
                        candidate_heur = heuristics_full[top_k_idx]
                    else:
                        candidate_idxs = feasible_idxs
                        candidate_heur = heuristics_full

                    # Scores: tau^alpha * eta^beta
                    tau_vals = np.array([self.pheromone[vm.id, j] for j in candidate_idxs], dtype=float)
                    scores = (tau_vals ** self.alpha) * (np.maximum(candidate_heur, 0.0) ** self.beta)

                    if scores.sum() <= 0 or np.all(np.isnan(scores)):
                        chosen_pm_idx = random.choice(candidate_idxs)
                    else:
                        # ACS choice: greedy with prob q0 else probabilistic
                        if random.random() < self.q0:
                            chosen_pm_idx = candidate_idxs[int(np.argmax(scores))]
                        else:
                            probs = scores / scores.sum()
                            chosen_pm_idx = np.random.choice(candidate_idxs, p=probs)

                    # Place and local pheromone update
                    if temp_pms[chosen_pm_idx].can_host(vm):
                        temp_pms[chosen_pm_idx].place_vm(vm)
                        solution_mapping[vm.id] = temp_pms[chosen_pm_idx].id
                        self._acs_local_update(vm.id, chosen_pm_idx)

                # Evaluate ant solution
                fitness = self._calculate_fitness(temp_pms)
                all_ant_solutions.append((solution_mapping, fitness))

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution_mapping = solution_mapping

            # Global pheromone update (ACS): evaporation + deposit on best of iteration
            self.pheromone *= (1.0 - self.rho)
            if all_ant_solutions:
                best_iter_solution, best_iter_fitness = min(all_ant_solutions, key=lambda x: x[1])
                delta_tau = 1.0 / (best_iter_fitness + 1e-9)
                for vm_id, pm_id in best_iter_solution.items():
                    self.pheromone[vm_id, pm_id] += self.rho * delta_tau

            # Enforce pheromone bounds
            np.clip(self.pheromone, self.tau_min, self.tau_max, out=self.pheromone)

            print(f"Iteration {i+1}/{self.n_iterations}: Best Fitness = {best_fitness:.6f}")

            # Early stopping
            if i == 0:
                prev_best = best_fitness
            else:
                if best_fitness + 1e-12 < prev_best:
                    no_improve = 0
                    prev_best = best_fitness
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        print("Early stopping: no improvement observed.")
                        break

        # Apply the best found mapping to the actual PMs
        for pm in self.pms:
            pm.reset()
        if best_solution_mapping:
            for vm_id, pm_id in best_solution_mapping.items():
                self.pms[pm_id].place_vm(self.vms[vm_id])
        return self.pms