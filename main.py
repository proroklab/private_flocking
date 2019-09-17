import os
import glob
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import pygmo as pg
from model import Network
import genetic_algo
from config import args

device = torch.device(args.device)

def main():
    seed = args.seed
    np.random.seed(seed)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)
    timestamp = str(utils.get_unix_timestamp())
    utils.makedirs(args.save)
    path = os.path.join(args.save, timestamp)
    utils.create_exp_dir(path, scripts_to_save=glob.glob('../*.py'))
    logger = utils.get_logger(args.save, timestamp, file_type='txt')
    utils.makedirs(os.path.join(path, 'logs'))
    logger.info("time = %s, args = %s", str(utils.get_unix_timestamp()), args)

    input_shape = [11, 9, 3] # MANUALLY SET NUMBER OF CHANNELS (11) ACCORDING TO PRETRAINING

    os.system('cp -f ../pretrain-weights.pt {}'.format(os.path.join(path, 'weights.pt')))
    utils.makedirs(os.path.join(path, 'scripts'))
    os.system('cp -f ./for-copy/parse-ga.py {}'.format(os.path.join(path, 'scripts', 'parse-ga.py')))
    os.system('cp -f ./for-copy/parse-ga.py {}'.format(os.path.join(path, 'scripts', 'parse-log.py')))
    os.system('cp -f ./for-copy/parse_data.py {}'.format(os.path.join(path, 'scripts', 'parse_data.py')))
    os.system('cp -f ./for-copy/optimization-plots.sh {}'.format(os.path.join(path, 'scripts', '1_optimization-plots.sh')))

    # PyTorch
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    model = Network(input_shape, args.num_drones, criterion, path)
    model = model.to(device)
    utils.load(model, os.path.join(path, 'weights.pt'))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    # PyGMO
    prob = pg.problem(genetic_algo.Flocking(path, timestamp, model))
    pop = pg.population(prob, size=10, seed=24601)
    algo = pg.algorithm(pg.sga(gen = 1, cr = .90, m = 0.02, param_s = 3,
                               crossover = "single", mutation = "uniform", selection = "truncated"))
    algo.set_verbosity(1)

    for i in range(29):
        logger.info("time = %s gen = %d \n champ_f = %s \n champ_x = %s \n f_s = %s \n x_s = %s \n id_s = %s",
            str(utils.get_unix_timestamp()),
            i + 1,
            str(np.array(pop.champion_f).tolist()),
            str(np.array(pop.champion_x).tolist()),
            str(np.array(pop.get_f()).tolist()),
            str(np.array(pop.get_x()).tolist()),
            str(np.array(pop.get_ID()).tolist()))
        pop = algo.evolve(pop)
        model.online_update(path, genetic_algo.TS_LIST[-100:], input_shape, criterion, optimizer, logger, i)
        utils.save(model, os.path.join(path, 'weights.pt'))


if __name__ == '__main__':
    main()
