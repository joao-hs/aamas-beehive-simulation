from bee_colonies.env.bee_colonies import BeeColonyEnv
from pygame import event, QUIT, quit
from time import sleep
import numpy as np

from bee_colonies.models.queen_bee import QueenBee

RANDOM = True
GREEDY = False
COLAB = False

SEED = 42
N_BEES_PER_COLONY = (45, 20)
N_WASPS = 50
FLOWER_PROB = 0.1
VISION = 2
MAX_STEPS = 1000

if __name__ == "__main__":
    env = BeeColonyEnv(seed=SEED, grid_shape=(100,100), n_wasps=N_WASPS, n_bees_per_colony=N_BEES_PER_COLONY, flower_density=FLOWER_PROB, range_of_vision=VISION, max_steps=MAX_STEPS)
    observations = env.reset()

    done = False
    masks = {
        agent.id: np.zeros(env.action_spaces[agent.id].n, dtype=np.int8) if not isinstance(agent, QueenBee) else 2*np.ones(env.action_spaces[agent.id].n, dtype=np.int8) for agent in env.agents
    }
    while not done:
        for e in event.get():
            if e.type == QUIT:
                quit()
                break
        print("Step", env.timestep)

        actions = {agent: env.action_space(agent.id).sample(mask=masks[agent.id]) for agent in env.agents}
        # print("Actions", actions)
        observations, rewards, masks, done, truncations = env.step(actions)
        env.render()
        print('-'*20)
    quit()
    env.close()