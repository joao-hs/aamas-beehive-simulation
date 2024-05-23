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
        observations, rewards, masks, done, info = env.step(actions)
        print(info)
        if done:
            doneFor += 1
        agents_observe(env, observations, masks)

        env.render()
        print('-' * 20)


def create_scenario(queen_bee_classes, bee_classes, wasp_class) -> BeeColonyEnv:
    queen_bees: list[QueenBee] = [
        queen_bee_classes[colony](
            id=colony,
            bees=[
                bee_classes[colony](
                    local_beehive_id=i, cluster_center_distance=MAX_DISTANCE_FROM_CLUSTER
                ) for i in range(N_BEES_PER_COLONY[colony])
            ],
            new_bee_class=bee_classes[colony],
            n_clusters=NUM_FLOWER_CLUSTERS,
            cluster_center_distance=MAX_DISTANCE_FROM_CLUSTER
        ) for colony in range(N_COLONIES)

    ]

    bees: tuple[list[Bee], ...] = tuple(
        queen_bee.bees for queen_bee in queen_bees
    )

    for colony, colony_bees in enumerate(bees):
        for bee in colony_bees:
            bee.set_queen(queen_bees[colony])

    wasps: list[Wasp] = [wasp_class(i, NUM_FLOWER_CLUSTERS, MAX_DISTANCE_FROM_CLUSTER) for i in range(N_WASPS)]

    env = BeeColonyEnv(queen_bees, bees, wasps, seed=SEED, grid_shape=(75, 75), n_wasps=N_WASPS,
                       n_bees_per_colony=N_BEES_PER_COLONY, flower_density=FLOWER_PROB,
                       num_clusters=NUM_FLOWER_CLUSTERS, max_distance_from_cluster=MAX_DISTANCE_FROM_CLUSTER,
                       range_of_vision=VISION, max_steps=MAX_STEPS)
    return env


def main():
    environments = [
        # create_scenario([ConservativeQueenBee, ConservativeQueenBee], [GreedyBee, GreedyBee], GreedyWasp),
        # create_scenario([ConsiderateQueenBee, ConsiderateQueenBee], [GreedyBee, GreedyBee], GreedyWasp),
        create_scenario([ConsiderateQueenBee, ConsiderateQueenBee], [SocialBee, SocialBee], GreedyWasp),
        create_scenario([ConsiderateQueenBee, ConsiderateQueenBee], [RespectfulBee, RespectfulBee], GreedyWasp),
    ]

    for env in environments:
        if FAIR_TESTING:
            configure_seed(env.seed)
        run_env(env)
        env.close()


if __name__ == "__main__":
    main()
    quit()
