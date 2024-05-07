from bee_colonies.env.bee_colonies import BeeColonyEnv
from pygame import event, QUIT, quit


if __name__ == "__main__":
    env = BeeColonyEnv(seed=42, grid_shape=(100,100), n_wasps=50, n_bees_per_colony=(45, 20), flower_density=0.1)
    observations = env.reset()

    done = False
    while not done:
        for e in event.get():
            if e.type == QUIT:
                quit()
                break
        print("Step", env.timestep)
        actions = {agent: env.action_space(agent.id).sample() for agent in env.agents}
        observations, rewards, done, truncations = env.step(actions)
        env.render()
        print('-'*20)
    quit()
    env.close()