from bee_colonies.agents.bee.greedy_bee import GreedyBee
from bee_colonies.agents.bee.respectful_bee import RespectfulBee
from bee_colonies.agents.bee.social_bee import SocialBee
from bee_colonies.agents.queen_bee.conservative_queen_bee import ConservativeQueenBee
from bee_colonies.agents.queen_bee.considerate_queen_bee import ConsiderateQueenBee
from bee_colonies.agents.queen_bee.greedy_queen_bee import GreedyQueenBee
from bee_colonies.agents.wasp.greedy_wasp import GreedyWasp
from bee_colonies.env.bee_colonies import BeeColonyEnv, configure_seed
from bee_colonies.models.agent import Agent
from pygame import event, QUIT, quit
import numpy as np

from bee_colonies.models.bee import Bee
from bee_colonies.models.queen_bee import QueenBee
from bee_colonies.models.wasp import Wasp
import pandas as pd

RANDOM = True
GREEDY = False
COLAB = False

SEED = 42
N_BEES_PER_COLONY = (45, 20)
N_COLONIES = len(N_BEES_PER_COLONY)
N_WASPS = 2
FLOWER_PROB = 0.1
VISION = 3
NUM_FLOWER_CLUSTERS = 2
MAX_DISTANCE_FROM_CLUSTER = 25
MAX_STEPS = 1000
TIMESTEPS_AFTER_DONE = 5
FAIR_TESTING = True


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


def run_env(env, filename):
    columns = ['timestep', 'alive_queen1', 'dead_queen1', 'food_queen1', 'health_queen1', 'presence_queen1',
               'alive_queen2', 'dead_queen2', 'food_queen2', 'health_queen2', 'presence_queen2']
    simulation_data = pd.DataFrame(columns=columns)
    
    observations = env.reset()
    masks = env.init_masks()
    agents_observe(env, observations, masks)

    doneFor = 0
    while doneFor < TIMESTEPS_AFTER_DONE:
        for e in event.get():
            if e.type == QUIT:
                break
        print("Step", env.timestep)

        actions = compute_actions(env)
        observations, rewards, masks, done, info = env.step(actions)
        print(info)

        new_row = {
            'timestep': info['timestep'],
            'alive_queen1': info['alive'][0],
            'dead_queen1': info['dead_count'][0],
            'food_queen1': info['food'][0],
            'health_queen1': info['health'][0],
            'presence_queen1': info['presence_in_beehive'][0],
            'alive_queen2': info['alive'][1],
            'dead_queen2': info['dead_count'][1],
            'food_queen2': info['food'][1],
            'health_queen2': info['health'][1],
            'presence_queen2': info['presence_in_beehive'][1]
        }
        simulation_data = simulation_data._append(new_row, ignore_index=True)

        if done:
            doneFor += 1
        agents_observe(env, observations, masks)
        env.render()
        print('-' * 20)
    
    # Use the filename parameter to save the DataFrame to a specific file
    simulation_data.to_csv(filename, index=False)



def create_scenario(queen_bee_classes, bee_classes, wasp_class) -> BeeColonyEnv:
    queen_bees: list[QueenBee] = [
        queen_bee_classes[colony](
            id=colony,
            bees=[
                bee_classes[colony](local_beehive_id=i) for i in range(N_BEES_PER_COLONY[colony])
            ],
            new_bee_class=bee_classes[colony]
        ) for colony in range(N_COLONIES)

    ]

    bees: tuple[list[Bee], ...] = tuple(
        queen_bee.bees for queen_bee in queen_bees
    )

    for colony, colony_bees in enumerate(bees):
        for bee in colony_bees:
            bee.set_queen(queen_bees[colony])

    wasps: list[Wasp] = [wasp_class(i) for i in range(N_WASPS)]

    env = BeeColonyEnv(queen_bees, bees, wasps, seed=SEED, grid_shape=(75, 75), n_wasps=N_WASPS,
                       n_bees_per_colony=N_BEES_PER_COLONY, flower_density=FLOWER_PROB,
                       num_clusters=NUM_FLOWER_CLUSTERS, max_distance_from_cluster=MAX_DISTANCE_FROM_CLUSTER,
                       range_of_vision=VISION, max_steps=MAX_STEPS)
    return env


def main():
    # scenario: ([queen_bee_class1, queen_bee_class2, ..., queen_bee_classN], [bee_class1, bee_class2, ..., bee_classN], wasp_class, filename)
    scenarios = [
        ([ConservativeQueenBee, ConservativeQueenBee], [GreedyBee, GreedyBee], GreedyWasp, 'data/conservative_greedy.csv'),
        ([ConsiderateQueenBee, ConsiderateQueenBee], [GreedyBee, GreedyBee], GreedyWasp, 'data/considerate_greedy.csv'),
        ([ConsiderateQueenBee, ConsiderateQueenBee], [SocialBee, SocialBee], GreedyWasp, 'data/considerate_social.csv'),
        ([ConsiderateQueenBee, ConsiderateQueenBee], [RespectfulBee, RespectfulBee], GreedyWasp, 'data/considerate_respectful.csv')
    ]

    for queen_bee_classes, bee_classes, wasp_class, filename in scenarios:
        env = create_scenario(queen_bee_classes, bee_classes, wasp_class)
        if FAIR_TESTING:
            configure_seed(env.seed)
        run_env(env, filename)
        env.close()


if __name__ == "__main__":
    main()
    quit()
