from .agent import Agent

class Wasp(Agent):
    def __init__(self, id, initial_health=10):
        super().__init__(id)
        self.name = str(id)
        self.health = 20
        self.is_alive = True

    def receive_damage(self, damage):
        """
        Method to apply damage to the wasp. It reduces the health by the damage amount.
        If the health drops to 0 or below, it marks the wasp as not alive.
        """
        self.health -= damage
        if self.health <= 0:
            self.health = 0  # Ensure health doesn't go negative.
            self.is_alive = False

    def see(self, observation):
        return observation
    
    def action(self) -> int:
        return super().action()

    def __repr__(self):
        """
        Represent the Wasp object with its ID. If the wasp is not alive, enclose its ID in brackets.
        """
        rep = f"W{self.name}"
        if not self.is_alive:
            rep = f"[{rep}]"
        return rep
