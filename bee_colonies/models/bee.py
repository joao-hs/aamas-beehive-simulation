from .agent import Agent

class Bee(Agent):
    def __init__(self, id):
        super().__init__(id)
        self.name=str(id)
        self.hunger = 0
        self.is_alive = True

    def set_spawn(self, spawn_location: tuple[int]):
        self.beehive = spawn_location
        return super().set_spawn(spawn_location)

    def eat(self):
        if not self.is_alive:
            raise Exception("Bee is dead")
        if self.hunger > 0:
            self.hunger -= 1
    
    def starve(self):
        if not self.is_alive:
            raise Exception("Bee is dead")
        self.hunger += 1

    def see(self, observation):
        return observation
    
    def action(self) -> int:
        return super().action()

    def __repr__(self):
        rep = f"B{self.name}<{self.hunger}>"
        if not self.is_alive:
            rep = f"[{rep}]"
        return rep
    