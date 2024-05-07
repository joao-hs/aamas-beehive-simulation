from .agent import Agent

class QueenBee(Agent):
    def __init__(self, id):
        super().__init__(id)
        self.name=str(id)
        self.is_alive = True

    def see(self, observation):
        return observation
    
    def action(self) -> int:
        return super().action()

    def __repr__(self):
        return f"Q{self.name}"
    