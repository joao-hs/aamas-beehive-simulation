TIME_TO_RESTORE_POLLEN = 5


class Flower:
    def __init__(self, position) -> None:
        self.pollen = True
        self.position = position
        self.counter = 0

    def collect_pollen(self) -> bool:
        if self.pollen:
            self.pollen = False
            return True
        return False

    def timestep(self):
        if self.pollen:
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= TIME_TO_RESTORE_POLLEN:
                self.pollen = True
                self.counter = 0
                print(f"Flower at {self.position} has restored its polen.")
