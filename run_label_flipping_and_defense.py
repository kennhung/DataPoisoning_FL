import os
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import replace_0_with_2
from federated_learning.utils import replace_5_with_3
from federated_learning.utils import replace_1_with_9
from federated_learning.utils import replace_4_with_6
from federated_learning.utils import replace_1_with_3
from federated_learning.utils import replace_6_with_0
from federated_learning.worker_selection import RandomSelectionStrategy
from server import run_exp

from defense import run_defense

if __name__ == '__main__':
    START_EXP_IDX = 3000
    NUM_EXP = 1
    NUM_POISONED_WORKERS = 5
    REPLACEMENT_METHOD = replace_1_with_9
    KWARGS = {
        "NUM_WORKERS_PER_ROUND": 25
    }

    args = Arguments(logger)

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        result = run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS,
                         KWARGS, RandomSelectionStrategy(), experiment_id)
        run_defense(args, result['models_folders'][0], range(
            1, 10), 'fc.weight', 1, result['poisoned_workers'], 'out/defense_results_' + str(experiment_id) + '.jpg')
