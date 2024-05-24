import sys
import os
from copy import copy

from config import read_config

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        CONFIG = read_config(config_path)
        # write config path as env variable
        os.environ["CONFIG_PATH"] = config_path
    else:
        print("Usage: python main.py <config_file_path>")

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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        CONFIG = read_config(config_path)
    else:
        print("Usage: python main.py <config_file_path>")

SEED = CONFIG["seed"]
N_BEES_PER_COLONY = tuple(CONFIG["n_bees_per_colony"])
N_COLONIES = len(N_BEES_PER_COLONY)
N_WASPS = CONFIG["n_wasps"]
FLOWER_PROB = CONFIG["flower_prob"]
VISION = CONFIG["vision"]
NUM_FLOWER_CLUSTERS = CONFIG["num_flower_clusters"]  # for uniform distribution set clusters to 0
MAX_DISTANCE_FROM_CLUSTER = CONFIG["max_distance_from_cluster"]
MAX_STEPS = CONFIG["max_steps"]
TIMESTEPS_AFTER_DONE = CONFIG["timesteps_after_done"]
FAIR_TESTING = CONFIG["fair_testing"]


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
    columns = ['timestep', 'alive_queen1', 'dead_queen1', 'food_queen1', 'health_queen1', 'presence_queen1']
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


def parse_classes():
    queen_bee_classes_names = CONFIG["queen_bee_classes"]
    queen_bee_classes = []
    for scenario in range(CONFIG["num_scenarios"]):
        scenario_classes = []
        for class_name in queen_bee_classes_names[scenario]:
            if class_name == "ConservativeQueenBee":
                scenario_classes.append(ConservativeQueenBee)
            elif class_name == "ConsiderateQueenBee":
                scenario_classes.append(ConsiderateQueenBee)
            elif class_name == "GreedyQueenBee":
                scenario_classes.append(GreedyQueenBee)
            else:
                print("Unknown queen bee:", class_name)
                quit()
        queen_bee_classes.append(copy(scenario_classes))

    bee_classes_names = CONFIG["bee_classes"]
    bee_classes = []
    for scenario in range(CONFIG["num_scenarios"]):
        scenario_classes = []
        for class_name in bee_classes_names[scenario]:
            if class_name == "GreedyBee":
                scenario_classes.append(GreedyBee)
            elif class_name == "RespectfulBee":
                scenario_classes.append(RespectfulBee)
            elif class_name == "SocialBee":
                scenario_classes.append(SocialBee)
            else:
                print("Unknown bee:", class_name)
                quit()
        bee_classes.append(copy(scenario_classes))

    wasp_class_names = CONFIG["wasp_class"]
    wasp_classes = []
    for scenario in range(CONFIG["num_scenarios"]):
        if wasp_class_names[scenario] == "GreedyWasp":
            wasp_classes.append(GreedyWasp)
        else:
            print("Unknown wasp:", wasp_class_names[scenario])
            quit()

    return queen_bee_classes, bee_classes, wasp_classes


def main():
    num_scenarios = CONFIG["num_scenarios"]
    queen_bee_classes, bee_classes, wasp_class = parse_classes()
    # scenario: ([queen_bee_class1, queen_bee_class2, ..., queen_bee_classN], [bee_class1, bee_class2, ..., bee_classN], wasp_class, filename)
    scenarios = [
        (queen_bee_classes[scenario], bee_classes[scenario], wasp_class[scenario], CONFIG["out_csv_path"][scenario])
        for scenario in range(num_scenarios)
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
