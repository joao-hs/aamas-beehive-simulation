from .agent import Agent

class Bee(Agent):
    def __init__(self, id, beehive_id, local_beehive_id):
        super().__init__(id)
        self.name = str(id)
        self.is_alive = True
        self.polen = False  # Indicates if the bee is carrying pollen.
        self.beehive_id = beehive_id
        self.local_beehive_id = local_beehive_id

    def set_spawn(self, spawn_location: tuple[int]):
        self.beehive_location = spawn_location
        return super().set_spawn(spawn_location)
    
    def see(self, observation):
        self.observation = observation
    
    def action(self) -> int:
        return super().action()

    def collect_pollen(self):
        """Bee collects pollen from a flower. Since flowers have infinite pollen, just toggle state."""
        if not self.polen:
            self.polen = True
            print(f"Bee {self.name} collected pollen and is now carrying it.")

    def drop_pollen(self):
        """Bee drops off pollen at the beehive."""
        if self.polen:
            self.polen = False
            print(f"Bee {self.name} dropped off pollen at the beehive.")

    def __repr__(self):
        rep = f"B{self.beehive_id}.{self.local_beehive_id}"
        if not self.is_alive:
            rep = f"[{rep}]"
        else:
            rep += "<>" if self.polen else ""
        return rep
