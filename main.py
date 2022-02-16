import os
import shutil
import argparse
import yaml
import logging

import run_lib
from utils import str2bool, dict2namespace


def main(args):
    # Create log directory
    os.makedirs(args.logdir, exist_ok=True)

    # Convert config type from dict to namespace
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    # Backup a copy of config in logdir
    shutil.copy(args.config, os.path.join(args.logdir, 'config.yml'))

    # Create a logger
    logfile = os.path.join(args.logdir, f'{args.mode}_log.txt')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=logfile)
    formatter = logging.Formatter(
        '%(filename)s - %(asctime)s - %(message)s'
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel('INFO')

    # Choose a pipeline
    if args.mode == 'train':
        run_lib.train(config, args.logdir, args.resume)
    elif args.mode == 'eval':
        run_lib.eval(config, args.logdir)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to the config file. See "./configs/".',
    )
    parser.add_argument(
        '--mode', type=str, required=True,
        help='Running mode. Should be "train" or "eval".',
    )
    parser.add_argument(
        '--logdir', type=str, default='./logs/any_name',
        help='Logging directory. Recommended to be "./logs/any_name".',
    )
    parser.add_argument(
        '--resume', type=str2bool, default=True,
        help='If true, resume the previous training.'
    )
    args = parser.parse_args()
    main(args)


    
