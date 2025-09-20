# file: components.py

class VM:
    """Represents a Virtual Machine with resource requirements."""
    def __init__(self, vm_id, cpu_req, mem_req):
        self.id = vm_id
        self.cpu = cpu_req
        self.mem = mem_req

class PM:
    """Represents a Physical Machine with resource capacity and a power model."""
    def __init__(self, pm_id, cpu_cap, mem_cap):
        self.id = pm_id
        self.cpu_cap = cpu_cap
        self.mem_cap = mem_cap
        self.vms = []  # List of VMs placed on this PM
        self.cpu_used = 0
        self.mem_used = 0
        # [cite_start]Power model parameters as per typical models [cite: 57, 69]
        self.power_idle = 150  # Watts for an idle server
        self.power_max = 300   # Watts at 100% utilization

    def can_host(self, vm):
        """Checks if the PM has enough resources to host a given VM."""
        return (self.cpu_cap - self.cpu_used >= vm.cpu and
                self.mem_cap - self.mem_used >= vm.mem)

    def place_vm(self, vm):
        """Places a VM on this PM and updates resource usage."""
        if self.can_host(vm):
            self.vms.append(vm)
            self.cpu_used += vm.cpu
            self.mem_used += vm.mem
            return True
        return False

    def get_cpu_utilization(self):
        """Calculates the current CPU utilization of the PM."""
        return self.cpu_used / self.cpu_cap if self.cpu_cap > 0 else 0

    def get_power_consumption(self):
        """
        Calculates the current power consumption based on CPU utilization.
        If no VMs are on the PM, it's considered powered off, consuming 0 watts.
        """
        if not self.vms:
            return 0
        util = self.get_cpu_utilization()
        return self.power_idle + (self.power_max - self.power_idle) * util

    def reset(self):
        """Resets the PM's state for a new simulation run."""
        self.vms = []
        self.cpu_used = 0
        self.mem_used = 0