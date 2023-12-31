# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Extracts Pareto-frontier through Evolutionary Search, given constraints. 
"""

import argparse
import pickle
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml
import shutil

from archai.common import utils
from archai.nlp.nas.evolution import Evolution
from archai.nlp.nas.nas_utils.constraints.constraint_pipeline import DEVICE_LATENCY_CONSTRAINT
from archai.nlp.nas.nas_utils.dispatcher import create_pareto_jobs
from archai.nlp.nas.baseline_utils import plot_baseline_and_pareto, profile_baseline, select_pareto, get_latex_tables


def parse_args():
    parser = argparse.ArgumentParser(description='Language models Pareto-frontier extraction.')
    
    baseline = parser.add_argument_group('NAS baseline profiling and plotting')
    baseline.add_argument('--profile_baseline',
                        action='store_true',
                        help='Measure proxy and hardware metrics for NAS baseline.')
    
    baseline.add_argument('--baseline_path',
                        type=str,
                        default='baseline_logs',
                        help='Path to the folder used to save baseline jobs.')

    baseline.add_argument('--select_pareto',
                          action='store_true',
                          help='Select a subset of the extracted pareto for full training, based on the baseline architectures.')

    baseline.add_argument('--plot_pareto_baseline',
                          action='store_true',
                          help='Print selected pareto points and baseline after full training.')
    
    baseline.add_argument('--gen_tables',
                            action='store_true',
                            help='generate latex tables from the pareto front models on a device')

    search = parser.add_argument_group('Search configuration')
    search.add_argument('--default_path',
                        type=str,
                        default='logdir',
                        help='Path to the default folder used to save outputs.')

    search.add_argument('--model_type',
                        type=str,
                        default='mem_transformer',
                        choices=['hf_gpt2', 'hf_gpt2_flex', 'hf_transfo_xl', 'mem_transformer','hf_gpt','hf_opt','hf_ctrl'],
                        help='Type of model to be searched.')

    search.add_argument('--model_config',
                        type=str,
                        default=None,
                        help='YAML configuration file to override default configuration.')

    search.add_argument('--population_size',
                        type=int,
                        default=100,
                        help='Size of the population.')

    search.add_argument('--parent_size',
                        type=int,
                        default=20,
                        help='Size of the parent genes.')

    search.add_argument('--mutation_size',
                        type=int,
                        default=40,
                        help='Size of the mutated genes.')
    
    search.add_argument('--mutation_prob',
                        type=float,
                        default=0.3,
                        help='Probability of mutation.')

    search.add_argument('--crossover_size',
                        type=int,
                        default=40,
                        help='Size of the crossovered genes.')

    search.add_argument('--crossover_prob',
                        type=float,
                        default=0.5,
                        help='Probability of crossover.')

    search.add_argument('--n_iter',
                        type=int,
                        default=10,
                        help='Number of search iterations.')

    search.add_argument('--do_brute_force',
                        action='store_true',
                        help='Uses brute force instead of standard search.')

    search.add_argument('--n_samples',
                        type=int,
                        default=20000,
                        help='Number of genes used to sample during brute force.')

    search.add_argument('--batch',
                        type=int,
                        default=1000,
                        help='Number of batched genes used to conduct the brute force.')

    search.add_argument('--use_quantization',
                        action='store_true',
                        help='Uses quantized models to conduct the search.')

    search.add_argument('--seed',
                        type=int,
                        default=1111,
                        help='Random seed.')

    strategy = parser.add_argument_group('Training strategy')

    strategy.add_argument('--training_strategy',
                          type=str,
                          default='decoder_params',
                          choices=['decoder_params', 'val_ppl', 'char_accept_rate'],
                          help='Training strategy: decoder parameters, validation perplexity or character accept rate.')

    strategy.add_argument('--dataset',
                          type=str,
                          default='wt103',
                          choices=['wt103', 'lm1b','wt2'],
                          help='Dataset (if not using `decoder_params`).')

    strategy.add_argument('--scoring_file',
                          type=str,
                          default=None,
                          help='Scoring .ljson file (if using `char_accept_rate`).')

    strategy.add_argument('--vocab_type',
                          type=str,
                          default='word',
                          choices=['word', 'bppe', 'gpt2'],
                          help='Type of vocabulary (if not using `decoder_params`).')

    strategy.add_argument('--vocab_size',
                          type=int,
                          default=10000,
                          help='Size of vocabulary (if not using `decoder_params`).')

    strategy.add_argument('--training_max_step',
                          type=int,
                          default=100,
                          help='Maximum number of training steps (if not using `decoder_params`).')

    choice = parser.add_argument_group('Hyperparameters choices')
    choice.add_argument('--n_layer',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Choices for number of layers.')

    choice.add_argument('--d_model',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Choices for model dimensions.')

    choice.add_argument('--d_inner',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Choices for inner dimensions.')

    choice.add_argument('--n_head',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Choices for number of attention heads.')

    choice.add_argument('--div_val',
                        nargs='+',
                        type=int,
                        default=None,
                        help='Choices for dividents.')

    constraint = parser.add_argument_group('Constraints')
    constraint.add_argument('--constraint_pipeline_type',
                            default='torch',
                            choices=['onnx', 'torch'],
                            help='Type of constraint pipeline to be used during search.')

    constraint.add_argument('--param_constraint_lower',
                            type=int,
                            default=5e6,
                            help='Candidates below total parameters will be rejected.')

    constraint.add_argument('--param_constraint_upper',
                            type=int,
                            default=12e6,
                            help='Candidates above total parameters will be rejected.')

    constraint.add_argument('--latency_constraint_upper',
                            type=float,
                            default=None,
                            help='Candidates above latency will be rejected.')

    constraint.add_argument('--n_threads',
                            type=int,
                            default=1,
                            help='Number of inference threads.')

    constraint.add_argument('--latency_repeat',
                            type=int,
                            default=10,
                            help='Number of latency measurements.')

    constraint.add_argument('--device',
                            type=str,
                            default='cpu',
                            help='Type of device (`cpu` or `cuda`).')

    constraint.add_argument('--device_name',
                            type=str,
                            default='XeonE5-2690',
                            help='Name of device that search is being conducted on.')

    constraint.add_argument('--eps',
                            type=float,
                            default=0.05,
                            help='Value for neighborhood used around the Pareto front.')
                        
    args, _ = parser.parse_known_args()

    return vars(args)


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Applies random seeds
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    if not torch.cuda.is_available():
        args['use_training_proxy'] = True
        print('No CUDA available, defaulting to `use_training_proxy` as True.')

    # Gathers the latency constraint based on device
    if args['latency_constraint_upper'] is None:
        args['latency_constraint_upper'] = DEVICE_LATENCY_CONSTRAINT[args['device_name']]

    # Initializes the result's path
    now = datetime.now()
    time_str = now.strftime('%d_%m_%Y_%H_%M_%S')
    results_path_str = f'{args["model_type"]}_lower_param_{args["param_constraint_lower"]/1e6}M_upper_param_{args["param_constraint_upper"]/1e6}M_latency_upper_{args["latency_constraint_upper"]}s_{args["device_name"]}_{time_str}'
    results_path = os.path.join(args['default_path'], results_path_str)
    args['results_path'] = utils.full_path(results_path, create=True)

    # Dumps the search configuration to a YAML file
    path_to_search_config = os.path.join(args['results_path'], 'search_config.yaml')
    with open(path_to_search_config, 'w') as f:
        yaml.dump(args, f)

    # Loads model configuration file (if provided)
    try:
        with open(args['model_config'], 'r') as f:
            args['model_config'] = yaml.load(f, Loader=yaml.Loader)['train']
    except:
        args['model_config'] = {}

    do_profile = False if (args['profile_baseline'] or args['select_pareto'] or args['plot_pareto_baseline']) or args['gen_tables'] else True

    # Creates the evolutionary search instance
    e = Evolution(args['results_path'],
                  model_type=args['model_type'],
                  model_config=args['model_config'],
                  population_size=args['population_size'],
                  parent_size=args['parent_size'],
                  mutation_size=args['mutation_size'],
                  mutation_prob=args['mutation_prob'],
                  crossover_size=args['crossover_size'],
                  crossover_prob=args['crossover_prob'],
                  n_iter=args['n_iter'],
                  use_quantization=args['use_quantization'],
                  training_strategy=args['training_strategy'],
                  dataset=args['dataset'],
                  scoring_file=args['scoring_file'],
                  vocab_type=args['vocab_type'],
                  vocab_size=args['vocab_size'],
                  training_max_step=args['training_max_step'],
                  constraint_pipeline_type=args['constraint_pipeline_type'],
                  param_constraint_lower=args['param_constraint_lower'],
                  param_constraint_upper=args['param_constraint_upper'],
                  latency_constraint_upper=args['latency_constraint_upper'],
                  n_threads=args['n_threads'],
                  latency_repeat=args['latency_repeat'],
                  n_layer=args['n_layer'],
                  d_model=args['d_model'],
                  d_inner=args['d_inner'],
                  n_head=args['n_head'],
                  div_val=args['div_val'],
                  device=args['device'],
                  do_profile=do_profile)
    
    if args['profile_baseline'] or args['select_pareto'] or args['plot_pareto_baseline'] or args['gen_tables']:
        path_to_baseline = os.path.join(args['baseline_path'], args['model_type'])
        
        if args['profile_baseline']:
            shutil.copy(path_to_search_config, path_to_baseline)
            shutil.rmtree(args['results_path'])
            
            baseline_logs = profile_baseline(e, path_to_results=path_to_baseline, dataset=args['dataset'], device_name=args['device_name'])            
        
        elif args['select_pareto']:
            shutil.rmtree(args['results_path'])
            
            search_experiment = '{}_{}_3d'.format(args['model_type'], args['device_name'])
            last_iter = 12 ######################################### CHANGE THIS TO THE LAST ITERATION OF THE SEARCH
            e.load_state(path_to_logs=os.path.join(args['default_path'], search_experiment, f'logs_iter_{last_iter}.pkl'))
            
            with open(os.path.join(path_to_baseline, args['device_name'],'logs.pkl'), 'rb') as f:
                baseline_logs = pickle.load(f)
            e.plot_search_state(last_iter+1, parents=None, baseline=baseline_logs)
            e = select_pareto(e, path_to_results=os.path.join(path_to_baseline, args['device_name']))
            e.plot_search_state('selected_pareto', parents=None, baseline=baseline_logs)

            exp_name = 'pareto_jobs_{}_{}'.format(args['dataset'], args['device_name'])
            job_path = os.path.join(args['default_path'], search_experiment, exp_name)
            create_pareto_jobs(os.path.join(path_to_baseline, args['device_name']), 
                               converter=e.converter,
                               model_type=e.model_type,
                               max_step=100000 if args['dataset']=='lm1b' else 40000,
                               dataset=args['dataset'],
                               vocab=args['vocab_type'],
                               vocab_size=None if args['vocab_type']=='word' else args['vocab_size'], 
                               n_jobs=10,
                               output_path=job_path)   

        elif args['plot_pareto_baseline']:
            shutil.rmtree(args['results_path'])
            model = args['model_type']
            # print(model)
            # search_experiment = os.path.join('/home/mojan/TransformerNAS/amlt_logs', 'pareto_{}_{}_{}_3d'.format(model, args['dataset'], args['device_name']))
            search_experiment = "/content/drive/Shareddrives/ECE 226 Project/archai-neurips-lts/archai-neurips-lts/logdir/hf_ctrl_titanxp_3d/pareto"
            plot_baseline_and_pareto(path_to_amlt_logs=search_experiment, path_to_baseline_logs=path_to_baseline,
                                    dataset=args['dataset'], device_name=args['device_name'])

        elif args['gen_tables']: # generate latex tables from the pareto front models on a device
            shutil.rmtree(args['results_path'])
            model = args['model_type']
            search_experiment = os.path.join('/home/mojan/TransformerNAS/amlt_logs', 'pareto_{}_{}_{}_3d'.format(model, args['dataset'], args['device_name']))
            get_latex_tables(path_to_baseline_logs=path_to_baseline, 
                            dataset=args['dataset'], device_name=args['device_name'])
        
        exit()

    # Runs the evolutionary search or the brute force version
    e.run(do_brute_force=args['do_brute_force'],
          n_samples=args['n_samples'],
          batch=args['batch'])
