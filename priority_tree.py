from typing import Tuple
import numpy as np

class PriorityTree:
    def __init__(self, capacity, prio_exponent, is_exponent):
        self.num_layers = 1
        while capacity > 2**(self.num_layers-1): 
            self.num_layers += 1

        self.ptree = np.zeros(2**self.num_layers-1, dtype=np.float64)

        self.prio_exponent = prio_exponent
        self.is_exponent = is_exponent
    
    def update(self, idxes: np.ndarray, td_error: np.ndarray):
        priorities = td_error ** self.prio_exponent

        idxes = idxes + 2**(self.num_layers-1) - 1
        self.ptree[idxes] = priorities

        for _ in range(self.num_layers-1):
            idxes = (idxes-1) // 2
            idxes = np.unique(idxes)
            self.ptree[idxes] = self.ptree[2*idxes+1] + self.ptree[2*idxes+2]

    def sample(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        p_sum = self.ptree[0]
        interval = p_sum / num_samples

        prefixsums = np.arange(0, p_sum, interval, dtype=np.float64) + np.random.uniform(0, interval, num_samples)

        idxes = np.zeros(num_samples, dtype=np.int64)
        for _ in range(self.num_layers-1):
            nodes = self.ptree[idxes*2+1]
            idxes = np.where(prefixsums < nodes, idxes*2+1, idxes*2+2)
            prefixsums = np.where(idxes%2 == 0, prefixsums - self.ptree[idxes-1], prefixsums)
        
        # importance sampling weights
        priorities = self.ptree[idxes]
        min_p = np.min(priorities)
        is_weights = np.power(priorities/min_p, -self.is_exponent)

        idxes -= 2**(self.num_layers-1) - 1

        return idxes, is_weights