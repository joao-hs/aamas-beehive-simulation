from .agent import Agent
import numpy as np

class QueenBee(Agent):
    def __init__(self, id, n_bees=0):
        super().__init__(id)
        self.name=str(id)
        self.is_alive = True
        self.presence_array = np.ones(n_bees)
        self.health = 100  

    def see(self, observation):
        return observation
    
    def action(self) -> int:
        return super().action()

    def __repr__(self):
        return f"Queen Bee {self.id} with health {self.health}"
    
    def receive_damage(self, damage):
        """Reduces the health of the Queen Bee by the specified damage amount."""
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            print(f"Queen Bee {self.id} has been destroyed.")
        else:
            print(f"Queen Bee {self.id} now has {self.health} health.")