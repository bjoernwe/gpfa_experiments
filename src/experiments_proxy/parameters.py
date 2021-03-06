import matplotlib.pyplot as plt
import mdp
import numpy as np

import explot as ep

#import experiments_proxy.experiment_base_proxy as eb
import experiment_base as eb

from envs import env_data
from envs import env_data2d
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_random import EnvRandom



default_args_global = {'n_train':      10000,
                       'n_test':       2000,
                       'seed':         0,
                       'output_dim':   range(1,6),
                       'output_dim_max': 5,
                       'noisy_dims':   0,
                       'limit_data':   100000,
                       'use_test_set': True}

default_args_explot = {'repetitions':  50,
                       'cachedir':     '/local/weghebvc',
                       'manage_seed':  'repetition_index',
                       'verbose':      True,
                       'processes':    None}

#default_args_low  = {#'pca':         1.,
                     #'output_dim':  range(1,6),
                     #'output_dim_max': 5,
                     #'repetitions': 10,
#                     }

#default_args_high = {#'pca':         .99,
                     #'output_dim':  range(1,11),
                     #'output_dim_max': 10, 
                     #'repetitions': 5,
#                     }

algorithm_measures = {eb.Algorithms.None:     None,
                      eb.Algorithms.PCA:      eb.Measures.delta,
                      eb.Algorithms.Random:   eb.Measures.delta,
                      eb.Algorithms.SFA:      eb.Measures.delta,
                      eb.Algorithms.SFFA:     eb.Measures.delta,
                      eb.Algorithms.ForeCA:   eb.Measures.omega,
                      eb.Algorithms.PFA:      eb.Measures.pfa,
                      eb.Algorithms.GPFA1:    eb.Measures.gpfa,
                      eb.Algorithms.GPFA2:    eb.Measures.gpfa,
                      #
                      eb.Algorithms.HiRandom: eb.Measures.delta,
                      eb.Algorithms.HiSFA:    eb.Measures.delta,
                      eb.Algorithms.HiPFA:    eb.Measures.pfa,
                      eb.Algorithms.HiGPFA:   eb.Measures.gpfa,
                      }

algorithm_args = {eb.Algorithms.None:   {},
                  eb.Algorithms.Random: {},
                  eb.Algorithms.SFA:    {},
                  eb.Algorithms.ForeCA: {'n_train':      1000, 
                                         'n_test':       200,
                                         #'pca':          1.,
                                         'output_dim':  range(1,6),
                                         'output_dim_max': 5,
                                         'repetitions':    10,
                                         },
                  eb.Algorithms.PFA:    {'repetitions':  50,},
                  eb.Algorithms.GPFA2:  {'iterations':   30,
                                         'k_eval':       10,
                                         'repetitions':  10,
                                         },
                  #eb.Algorithms.HiSFA:  {'output_dim':     5,
                  #                       'output_dim_max': 5,},
                  #eb.Algorithms.HiPFA:  {'output_dim':   5,},
                  #eb.Algorithms.HiGPFA: {'output_dim':   5,
                  #                       'iterations':   30}
                  }

dataset_args = [{'env': EnvData, 'dataset': env_data.Datasets.STFT1, 'pca': .99},
                {'env': EnvData, 'dataset': env_data.Datasets.STFT2, 'pca': .99},
                {'env': EnvData, 'dataset': env_data.Datasets.STFT3, 'pca': .99},

                #{'env': EnvData, 'dataset': env_data.Datasets.Spectro1, 'pca': .99},
                #{'env': EnvData, 'dataset': env_data.Datasets.Spectro2, 'pca': .99},
                #{'env': EnvData, 'dataset': env_data.Datasets.Spectro3, 'pca': .99},

                {'env': EnvData, 'dataset': env_data.Datasets.EEG2, 'pca': 1.},
                {'env': EnvData, 'dataset': env_data.Datasets.EEG, 'pca': 1.},
                {'env': EnvData, 'dataset': env_data.Datasets.EIGHT_EMOTION, 'pca': 1., 'n_train': 1000, 'n_test': 200},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG, 'pca': 1.},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH, 'pca': 1.},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG, 'pca': .99},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD, 'pca': 1.},
                    
                {'env': EnvData2D, 'dataset': env_data2d.Datasets.GoProBike,     'window': ((25,70),(45,90)), 'pca': .99},
                #{'env': EnvData2D, 'dataset': env_data2d.Datasets.GoProBike,     'window': ((70,70),(90,90)), 'pca': .99},
                #{'env': EnvData2D, 'dataset': env_data2d.Datasets.RatLab,        'window': ((0, 140), (40, 180)), 'scaling': .5, 'pca': .99},

                {'env': EnvData2D, 'dataset': env_data2d.Datasets.SpaceInvaders, 'window': ((16,30),(36,50)), 'pca': .99},
                {'env': EnvData2D, 'dataset': env_data2d.Datasets.Mario,         'window': ((70,70),(90,90)), 'pca': .99},
                {'env': EnvData2D, 'dataset': env_data2d.Datasets.Traffic,       'window': ((35,65),(55,85)), 'pca': .99},
    
                {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS, 'pca': 1., 'n_train': 1000, 'n_test': 200},
                {'env': EnvData, 'dataset': env_data.Datasets.HAPT, 'pca': .99, 'n_train': 5000, 'n_test': 1000},
                {'env': EnvRandom, 'dataset': None, 'ndim': 20, 'pca': 1},#, 'K': 0, 'p': 1, 'k': 1},
                ]

# extracting 10 dimensions when dim >= 20, extracting 5 otherwise
# dataset_default_args = {env_data.Datasets.PHYSIO_MGH: default_args_low,
#                         env_data.Datasets.PHYSIO_EHG: default_args_low,
#                         env_data.Datasets.PHYSIO_UCD: default_args_low,
#                         env_data.Datasets.EIGHT_EMOTION: default_args_high,
#                         env_data.Datasets.FIN_EQU_FUNDS: default_args_low,
#                         env_data.Datasets.EEG: default_args_high,
#                         env_data.Datasets.EEG2: default_args_high,
#                         None: default_args_low,
#                         env_data.Datasets.HAPT: default_args_high,
#                         env_data.Datasets.PHYSIO_MMG: default_args_low,
#                         env_data.Datasets.STFT1: default_args_high,
#                         env_data.Datasets.STFT2: default_args_high,
#                         env_data.Datasets.STFT3: default_args_high,
#                         env_data2d.Datasets.Mario: default_args_high,
#                         env_data2d.Datasets.Traffic: default_args_high,
#                         env_data2d.Datasets.SpaceInvaders: default_args_high}

# datasets with dim < 50
datasets_for_foreca = set([env_data.Datasets.PHYSIO_MGH,
                          env_data.Datasets.PHYSIO_EHG,
                          env_data.Datasets.PHYSIO_UCD,
                          env_data.Datasets.EIGHT_EMOTION,
                          env_data.Datasets.FIN_EQU_FUNDS,
                          env_data.Datasets.EEG,
                          #env_data.Datasets.EEG2,
                          #
                          ##env_data.Datasets.HAPT,
                          env_data.Datasets.PHYSIO_MMG,
                          ##env_data.Datasets.STFT1,
                          #env_data.Datasets.STFT2,
                          ##env_data.Datasets.STFT3,
                          ##env_data2d.Datasets.Mario,
                          #env_data2d.Datasets.Traffic,
                          env_data2d.Datasets.SpaceInvaders,
                          None])

# results from grid-search
algorithm_parameters = {eb.Algorithms.PFA: {env_data.Datasets.STFT1: {'p': 10, 'K': 10},
                                            env_data.Datasets.STFT2: {'p': 10, 'K': 10},
                                            env_data.Datasets.STFT3: {'p': 8, 'K': 1},
                                            env_data.Datasets.Spectro1: {'p': 4, 'K': 0},
                                            env_data.Datasets.Spectro2: {'p': 10, 'K': 1},
                                            env_data.Datasets.Spectro3: {'p': 6, 'K': 0},
                                            env_data.Datasets.EEG2: {'p': 10, 'K': 0},
                                            env_data.Datasets.EEG: {'p': 10, 'K': 1},
                                            env_data.Datasets.EIGHT_EMOTION: {'p': 8, 'K': 0},
                                            env_data.Datasets.PHYSIO_EHG: {'p': 10, 'K': 2},
                                            env_data.Datasets.PHYSIO_MGH: {'p': 10, 'K': 0},
                                            env_data.Datasets.PHYSIO_MMG: {'p': 2, 'K': 0},
                                            env_data.Datasets.PHYSIO_UCD: {'p': 8, 'K': 0},
                                            env_data2d.Datasets.GoProBike: {'p': 2, 'K': 0},
                                            env_data2d.Datasets.SpaceInvaders: {'p': 1, 'K': 0},
                                            env_data2d.Datasets.Mario: {'p': 2, 'K': 0},
                                            env_data2d.Datasets.Traffic: {'p': 1, 'K': 1},
                                            env_data.Datasets.FIN_EQU_FUNDS: {'p': 10, 'K': 10},
                                            env_data.Datasets.HAPT: {'p': 10, 'K': 1},
                                            None: {'p': 1, 'K': 10}},
                        eb.Algorithms.GPFA2: {env_data.Datasets.STFT1: {'p': 4, 'k': 10},
                                              env_data.Datasets.STFT2: {'p': 4, 'k': 1},
                                              env_data.Datasets.STFT3: {'p': 6, 'k': 1},
                                              env_data.Datasets.Spectro1: {'p': 2, 'k': 10},
                                              env_data.Datasets.Spectro2: {'p': 2, 'k': 2},
                                              env_data.Datasets.Spectro3: {'p': 1, 'k': 2},
                                              env_data.Datasets.EEG2: {'p': 1, 'k': 2},
                                              env_data.Datasets.EEG: {'p': 6, 'k': 5},
                                              env_data.Datasets.EIGHT_EMOTION: {'p': 1, 'k': 5},
                                              env_data.Datasets.PHYSIO_EHG: {'p': 4, 'k': 5},
                                              env_data.Datasets.PHYSIO_MGH: {'p': 1, 'k': 1},
                                              env_data.Datasets.PHYSIO_MMG: {'p': 1, 'k': 10},
                                              env_data.Datasets.PHYSIO_UCD: {'p': 1, 'k': 1},
                                              env_data2d.Datasets.GoProBike: {'p': 1, 'k': 1},
                                              env_data2d.Datasets.SpaceInvaders: {'p': 2, 'k': 10},
                                              env_data2d.Datasets.Mario: {'p': 1, 'k': 1},
                                              env_data2d.Datasets.Traffic: {'p': 2, 'k': 1},
                                              env_data.Datasets.FIN_EQU_FUNDS: {'p': 2, 'k': 10},
                                              env_data.Datasets.HAPT: {'p': 1, 'k': 10},
                                              None: {'p': 1, 'k': 2}},
                        eb.Algorithms.HiPFA: {env_data2d.Datasets.GoProBike: {'p': 4, 'K': 0},
                                              env_data2d.Datasets.SpaceInvaders: {'p': 2, 'K': 0},
                                              env_data2d.Datasets.Mario: {'p': 2, 'K': 0},
                                              env_data2d.Datasets.Traffic: {'p': 2, 'K': 0}},
                        eb.Algorithms.HiGPFA:{env_data2d.Datasets.GoProBike: {'p': 1, 'k': 1},
                                              env_data2d.Datasets.SpaceInvaders: {'p': 2, 'k': 2},
                                              env_data2d.Datasets.Mario: {'p': 2, 'k': 2},
                                              env_data2d.Datasets.Traffic: {'p': 2, 'k': 2}},

}



def get_results(alg, overide_args={}, include_random=True, plot=False):

    results = {}
    
    for args in dataset_args:
        if plot:
            plt.figure()
        env = args['env']
        dataset = args['dataset']
        print dataset
        if alg is eb.Algorithms.ForeCA and dataset not in datasets_for_foreca:
            continue
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
        if plot:
            ep.plot(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window', 'scaling'], show_plot=False, **kwargs)
        
    return results



def get_signals(alg, overide_args={}, include_random=True, stack_result=True, dataset_list=None):

    results = {}

    for args in dataset_args:
        env = args['env']
        dataset = args['dataset']
        if dataset_list is not None:
            if dataset not in dataset_list:
                continue
        print dataset
        if alg is eb.Algorithms.ForeCA and dataset not in datasets_for_foreca:
            continue
        if not include_random and env is EnvRandom:
            continue
        kwargs = dict(default_args_global)
        kwargs['algorithm'] = alg
        kwargs['measure'] = algorithm_measures[alg]
        kwargs.update(args)
        #kwargs.update(dataset_default_args.get(dataset, {}))
        kwargs.update(algorithm_parameters.get(alg, {}).get(dataset, {}))
        kwargs.update(algorithm_args.get(alg, {}))
        kwargs.update(overide_args)
    
        try:
            # list of repetition indices?
            projected_data_list = []
            data_train_list = []
            data_test_list = []
            model_list = []
            env_list = []
            for i in kwargs['seed']:
                print i, kwargs
                kwargs_updated = dict(kwargs)
                kwargs_updated['seed'] = i
                projected_data, model, [data_train, data_test], env_node = eb.calc_projected_data(**kwargs_updated)
                # lists of results
                projected_data_list.append(projected_data)
                data_train_list.append(data_train)
                data_test_list.append(data_test)
                model_list.append(model)
            if stack_result:
                projected_data = np.stack(projected_data_list, axis=2)
            else:
                projected_data = projected_data_list
            result = {'projected_data': projected_data, 'data_train': data_train_list, 'data_test': data_test_list,
                      'model': model_list, 'env_node': env_list}
        except TypeError:
            projected_data, model, [data_train, data_test], env_node = eb.calc_projected_data(**kwargs)
            result = {'projected_data': projected_data, 'data_train': data_train, 'data_test': data_test, 'model': model, 'env_node': env_node}
        results[dataset] = result
                
    return results


if __name__ == '__main__':
    print get_results(eb.Algorithms.SFA)
