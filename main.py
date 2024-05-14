from bee_colonies.env.bee_colonies import BeeColonyEnv
from pygame import event, QUIT, quit
from time import sleep
import numpy as np

from bee_colonies.models.queen_bee import QueenBee

if __name__ == "__main__":
    env = BeeColonyEnv(seed=42, grid_shape=(100,100), n_wasps=50, n_bees_per_colony=(45, 20), flower_density=0.1)
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
        print("Actions", actions)
        observations, rewards, masks, done, truncations = env.step(actions)
        env.render()
        print('-'*20)
    quit()
    env.close()