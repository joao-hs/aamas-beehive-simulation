from .agent import Agent
import numpy as np

class QueenBee(Agent):
    def __init__(self, id, n_bees=0):
        super().__init__(id)
        self.name=str(id)
        self.is_alive = True
        self.presence_array = np.ones(n_bees)

    def see(self, observation):
        return observation
    
    def action(self) -> int:
        return super().action()

    def __repr__(self):
        return f"Q{self.name}"
    