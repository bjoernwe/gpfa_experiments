import mdp
import numpy as np
#import sys

import explot as ep

import experiment_base as eb

from envs import env_data
from envs import env_data2d
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_random import EnvRandom

#sys.path.append('/home/weghebvc/PycharmProjects/gpfa/src/')
import random_node
import foreca.foreca_node as foreca_node
import PFANodeMDP
import gpfa_node


default_args_global = {'n_train':      10000, 
                       'n_test':       2000,
                       'output_dim':  range(1,6),
                       'seed':         0,
                       'noisy_dims':   0,
                       'limit_data':   100000,
                       'use_test_set': True,
                       #'expansion':    1
                       }

default_args_explot = {'repetitions':  20,
                       'cachedir':     '/local/weghebvc',
                       'manage_seed':  'repetition_index',
                       'verbose':      True,
                       'processes':    None}

#default_args_low  = {#'pca':         1.,
#                     'output_dim':  range(1,6),
#                     #'output_dim_max': 5,
#                     }

algorithm_measures = {eb.Algorithms.HiRandom: eb.Measures.delta,
                      eb.Algorithms.HiSFA:    eb.Measures.delta,
                      eb.Algorithms.HiSFFA:   eb.Measures.delta,
                      eb.Algorithms.HiPFA:    eb.Measures.pfa,
                      eb.Algorithms.HiGPFA:   eb.Measures.gpfa,
                      }

algorithm_args = {eb.Algorithms.HiRandom: {},
                  eb.Algorithms.HiSFA:    {},
                  eb.Algorithms.HiSFFA:   {},
                  eb.Algorithms.HiPFA:    {},
                  eb.Algorithms.HiGPFA:   {'iterations': 30,
                                           'k_eval': 10}
                  }

dataset_args_hi = [{'env': EnvData2D, 'dataset': env_data2d.Datasets.GoProBike,     'scaling': (50,50), 'window': ((0,45),( 90,115)), 'pca': 1., 'whitening': False},
                   {'env': EnvData2D, 'dataset': env_data2d.Datasets.SpaceInvaders, 'scaling': (50,50), 'window': ((0,14),( 52, 66)), 'pca': 1., 'whitening': False},
                   {'env': EnvData2D, 'dataset': env_data2d.Datasets.Mario,         'scaling': (50,50), 'window': ((0,20),(120,140)), 'pca': 1., 'whitening': False},
                   {'env': EnvData2D, 'dataset': env_data2d.Datasets.Traffic,       'scaling': (50,50), 'window': ((0,30),( 90,120)), 'pca': 1., 'whitening': False},
                   #{'env': EnvRandom, 'dataset': None, 'ndim': 2500, 'pca': 1},
                   ]

# extracting 10 dimensions when dim >= 20, extracting 5 otherwise
#dataset_default_args = {env_data2d.Datasets.Mario: default_args_low,
#                        env_data2d.Datasets.Traffic: default_args_low,
#                        env_data2d.Datasets.SpaceInvaders: default_args_low}

# results from grid-search
algorithm_parameters = {eb.Algorithms.HiPFA: {env_data2d.Datasets.GoProBike: {'p': 2, 'K': 0},
                                              env_data2d.Datasets.SpaceInvaders: {'p': 1, 'K': 0},
                                              env_data2d.Datasets.Mario: {'p': 2, 'K': 0},
                                              env_data2d.Datasets.Traffic: {'p': 1, 'K': 1},
                                              None: {'p': 1, 'K': 10}},
                        eb.Algorithms.HiGPFA:{env_data2d.Datasets.GoProBike: {'p': 1, 'k': 1},
                                              env_data2d.Datasets.SpaceInvaders: {'p': 2, 'k': 10},
                                              env_data2d.Datasets.Mario: {'p': 1, 'k': 1},
                                              env_data2d.Datasets.Traffic: {'p': 2, 'k': 1},
                                              None: {'p': 1, 'k': 2}}}



def get_results(alg, overide_args={}, include_random=True):

    results = {}
    
    for args in dataset_args_hi:
        env = args['env']
        dataset = args['dataset']
        print dataset
        if not include_random and env is EnvRandom:
            continue
        kwargs = dict(default_args_global)
        kwargs.update(default_args_explot)
        kwargs['algorithm'] = alg
        kwargs['measure'] = algorithm_measures[alg]
        kwargs.update(args)
        #kwargs.update(dataset_default_args.get(dataset, {}))
        kwargs.update(algorithm_parameters.get(alg, {}).get(dataset, {}))
        kwargs.update(algorithm_args.get(alg, {}))
        kwargs.update(overide_args)

        results[dataset] = ep.evaluate(eb.prediction_error, argument_order=['output_dim', 'principal_angle_idx'], ignore_arguments=['window', 'scaling'], **kwargs)
    return results



def get_signals(alg, overide_args={}, include_random=True):

    results = {}

    for args in dataset_args_hi:
        env = args['env']
        dataset = args['dataset']
        print dataset
        if not include_random and env is EnvRandom:
            continue
        kwargs = dict(default_args_global)
        kwargs['algorithm'] = alg
        #kwargs['measure'] = algorithm_measures[alg]
        kwargs.update(args)
        #kwargs.update(dataset_default_args.get(dataset, {}))
        kwargs.update(algorithm_parameters.get(alg, {}).get(dataset, {}))
        kwargs.update(algorithm_args.get(alg, {}))
        #kwargs.update({'output_dim': 5, 'output_dim_max': 5})
        kwargs.update(overide_args)
    
        try:
            # list of repetition indices?
            projected_data_list = []
            for i in kwargs['seed']:
                kwargs_updated = dict(kwargs)
                kwargs_updated['seed'] = i
                projected_data, _, [_, _], _ = eb.calc_projected_data(**kwargs_updated)
                projected_data_list.append(projected_data)
            projected_data = np.stack(projected_data_list, axis=2)
            data_train     = None
            data_test      = None
        except TypeError:
            projected_data, _, [data_train, data_test], _ = eb.calc_projected_data(**kwargs)
        result = {'projected_data': projected_data, 'data_train': data_train, 'data_test': data_test}
        results[dataset] = result

    return results



if __name__ == '__main__':
    print get_results(eb.Algorithms.SFA)

