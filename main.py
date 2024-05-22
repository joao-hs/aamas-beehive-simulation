from bee_colonies.agents.bee.greedy_bee import GreedyBee
from bee_colonies.env.bee_colonies import BeeColonyEnv
from bee_colonies.models.agent import Agent
from pygame import event, QUIT, quit
import numpy as np

from bee_colonies.models.bee import Bee
from bee_colonies.models.queen_bee import QueenBee
from bee_colonies.models.wasp import Wasp

RANDOM = True
GREEDY = False
COLAB = False

SEED = 42
N_BEES_PER_COLONY = (45, 20)
N_COLONIES = len(N_BEES_PER_COLONY)
N_WASPS = 50
FLOWER_PROB = 0.1
VISION = 3
MAX_STEPS = 1000
TIMESTEPS_AFTER_DONE = 5


def agents_observe(env, observations, masks):
    queen_bees_obs, bees_obs, wasps_obs = observations
    for queen_bee in env.queen_bees:
        queen_bee.see(queen_bees_obs[queen_bee.id], mask=masks[0][queen_bee.id])
    for colony, colony_bees in enumerate(env.bees_by_colony):
        for bee in colony_bees:
            bee.see(bees_obs[colony][bee.local_beehive_id], mask=masks[1][colony][bee.local_beehive_id])
    for wasp in env.wasps:
        wasp.see(wasps_obs[wasp.id], mask=masks[2][wasp.id])


def compute_actions(env):
    actions = {
        queen_bee: queen_bee.action() for queen_bee in env.queen_bees
    }
    actions.update({
        bee: bee.action() for colony_bees in env.bees_by_colony for bee in colony_bees
    })
    actions.update({
        wasp: wasp.action() for wasp in env.wasps
    })
    return actions


def run_env(env):
    observations = env.reset()

    masks = env.init_masks()
    agents_observe(env, observations, masks)

    doneFor = 0
    while doneFor < TIMESTEPS_AFTER_DONE:
        for e in event.get():
            if e.type == QUIT:
                break
        print("Step", env.timestep)

        actions: dict[Agent, int | np.ndarray] = compute_actions(env)
        observations, rewards, masks, done, truncations = env.step(actions)
        if done:
            doneFor += 1
        agents_observe(env, observations, masks)

        env.render()
        print('-' * 20)


def main():
    queen_bees: list[QueenBee] = [
        QueenBee(id=colony, bees=[
            GreedyBee(local_beehive_id=i) for i in range(N_BEES_PER_COLONY[colony])
        ]) for colony in range(N_COLONIES)
    ]

    bees: tuple[list[Bee], ...] = tuple(
        queen_bee.bees for queen_bee in queen_bees
    )

    for colony, colony_bees in enumerate(bees):
        for bee in colony_bees:
            bee.set_queen(queen_bees[colony])

    wasps: list[Wasp] = [Wasp(i) for i in range(N_WASPS)]

    env = BeeColonyEnv(queen_bees, bees, wasps, seed=SEED, grid_shape=(100, 100), n_wasps=N_WASPS,
                       n_bees_per_colony=N_BEES_PER_COLONY, flower_density=FLOWER_PROB,
                       range_of_vision=VISION, max_steps=MAX_STEPS)
    run_env(env)
    # print(queen_bees[0].received)
    env.close()


if __name__ == "__main__":
    main()
    quit()
