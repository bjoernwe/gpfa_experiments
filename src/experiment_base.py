"""
The functions in this file mainly have two purposes:

1) To provide a common interface to all the experiments, no matter what algorithm or dataset.
2) To allow for fine-grained caching of intermediate results through a modularized structure (with joblib).
"""

import joblib
import mdp
import numpy as np
#import sys

from enum import Enum

#sys.path.append('/home/weghebvc/PycharmProjects/gpfa/src/')
import foreca.foreca_node as foreca_node
import random_node
import gpfa_node
#import sfa_node
from utils import echo, f_identity, f_exp08, principal_angles

#sys.path.append('/home/weghebvc/workspace/git/explot/src/')
#import explot as ep

#sys.path.append('/home/weghebvc/PycharmProjects/pfa/src/')
import PFANodeMDP
import PFACoreUtil

#sys.path.append('/home/weghebvc/PycharmProjects/environments/src/')
#from envs.environment import Noise
from envs import env_data
from envs import env_data2d
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_predictable_noise import EnvPredictableNoise
from envs.env_random import EnvRandom


# prepare joblib.Memory
default_cachedir = '/local/weghebvc'
#default_cachedir = '/scratch/weghebvc'
#default_cachedir = '/home/weghebvc'
#default_cachedir = None
mem = joblib.Memory(cachedir=default_cachedir, verbose=1)


Algorithms = Enum('Algorithms', 'None PCA Random SFA SFFA ForeCA PFA GPFA1 GPFA2 HiRandom HiSFA HiSFFA HiPFA HiGPFA')

Measures = Enum('Measures', 'delta delta_ndim omega omega_ndim pfa gpfa gpfa_ndim ndims angle_to_sfa_signals angle_to_p1 min_deltas')



@echo
def prediction_error(measure, env, dataset, algorithm, output_dim, n_train, n_test,
                     use_test_set, seed, **kwargs):
    """
    This is the main function to be called from elsewhere. It allows for a complete parametrization of the experiments
    (algorithm, dataset, parameters, number of repetitions, etc.).

    :param measure:
    :param env:
    :param dataset:
    :param algorithm:
    :param output_dim:
    :param n_train:
    :param n_test:
    :param use_test_set:
    :param seed:
    :param kwargs:
    :return:
    """
    # rev: 12
    assert kwargs.get('repetition_index', None) is None
    projected_data, model, [data_train, data_test], _ = calc_projected_data(env=env,
                                                                         dataset=dataset,
                                                                         algorithm=algorithm,
                                                                         output_dim=output_dim,
                                                                         n_train=n_train,
                                                                         n_test=n_test,
                                                                         use_test_set=use_test_set,
                                                                         # repetition_index=repetition_index,
                                                                         seed=seed, **kwargs)

    kwargs.update({'env': env,
                   'dataset': dataset,
                   'algorithm': algorithm,
                   'output_dim': output_dim,
                   'n_train': n_train,
                   'n_test': n_test,
                   'use_test_set': use_test_set,
                   # 'repetition_index': repetition_index,
                   'seed': seed})
    error = prediction_error_on_data(data=projected_data, measure=measure, model=model,
                                     data_chunks=[data_train, data_test], **kwargs)
    #assert np.isfinite(error)
    return error


@echo
def calc_projected_data(env, dataset, algorithm, output_dim, n_train, n_test,  # repetition_index,
                        seed, noisy_dims=0, use_test_set=True, **kwargs):
    """
    This function also parameterizes a whole experiment but returns the learned signals. Useful for plotting, for
    instance.

    :param env:
    :param dataset:
    :param algorithm:
    :param output_dim:
    :param n_train:
    :param n_test:
    :param seed:
    :param noisy_dims:
    :param use_test_set:
    :param kwargs:
    :return:
    """
    if algorithm == Algorithms.PCA:
        kwargs['pca'] = output_dim
    [data_train, data_test], env_node = generate_training_data(env=env,
                                                               dataset=dataset,
                                                               n_train=n_train,
                                                               n_test=n_test,
                                                               noisy_dims=noisy_dims,
                                                               # repetition_index=repetition_index,
                                                               seed=seed,
                                                               **kwargs)
    print '%s: %d dimensions\n' % (dataset, data_train.shape[1])

    model = train_model(algorithm=algorithm,
                        data_train=data_train,
                        output_dim=output_dim,
                        seed=seed,
                        # repetition_index=repetition_index,
                        **kwargs)

    if model is None:
        if use_test_set:
            projected_data = np.array(data_test, copy=True)
        else:
            projected_data = np.array(data_train, copy=True)
    else:
        if use_test_set:
            projected_data = model.execute(data_test)
        else:
            projected_data = model.execute(data_train)
        # reduce dim because ForeCA calculated output_dim_max dimensions
        if algorithm == Algorithms.ForeCA:  # or \
            # algorithm == Algorithms.HiSFA:
            # algorithm == Algorithms.SFA or \
            # algorithm == Algorithms.SFFA or \
            projected_data = projected_data[:, :output_dim]

    return projected_data, model, [data_train, data_test], env_node


@echo
def prediction_error_on_data(data, measure, model=None, data_chunks=None, **kwargs):
    if data.ndim == 1:
        n = data.shape[0]
        data = np.array(data, ndmin=2).T
        assert data.shape == (n, 1)

    if measure == Measures.delta:
        return calc_delta(data=data, ndim=False)
    elif measure == Measures.delta_ndim:
        return calc_delta(data=data, ndim=True)
    elif measure == Measures.omega:
        return calc_omega(data=data, omega_dim=kwargs['output_dim'] - 1)
    elif measure == Measures.omega_ndim:
        return calc_omega_ndim(data=data)
    elif measure == Measures.pfa:
        return calc_autoregressive_error(data=data, model=model, p=kwargs['p'], data_train=data_chunks[0])
    # elif measure == Measures.pfa_ndim:
    #         return calc_autoregressive_error_ndim(data=data,
    #                                          p=kwargs['p'],
    #                                          K=kwargs['K'],
    #                                          model=model,
    #                                          data_chunks=data_chunks)
    elif measure == Measures.gpfa:
        return calc_predictability_trace_of_avg_cov(x=data,
                                                    k=kwargs.get('k_eval', kwargs['k']),
                                                    p=kwargs['p'],
                                                    ndim=False)
    elif measure == Measures.gpfa_ndim:
        return calc_predictability_trace_of_avg_cov(x=data,
                                                    k=kwargs.get('k_eval', kwargs['k']),
                                                    p=kwargs['p'],
                                                    ndim=True)
    elif measure == Measures.ndims:
        return data_chunks[0].shape[1]
    elif measure == Measures.angle_to_sfa_signals:
        return calc_angle_to_sfa_signals(data, **kwargs)
    elif measure == Measures.angle_to_p1:
        return calc_angle_to_p1(data, **kwargs)
    elif measure == Measures.min_deltas:
        dat = data_chunks[1] if kwargs['use_test_set'] else data_chunks[0]
        return calc_min_delta_components(dat, output_dim=kwargs['output_dim'])
    else:
        print measure
        assert False


@echo
def generate_training_data(env, dataset, n_train, n_test, seed, **kwargs):
    """
    A common interface to generate all the different datasets.

    :param env:
    :param dataset:
    :param n_train:
    :param n_test:
    :param seed:
    :param kwargs:
    :return:
    """

    if env is EnvData:
        env_node = EnvData(dataset=dataset, limit_data=kwargs['limit_data'], sampling_rate=kwargs.get('sampling_rate', 1), seed=seed)
    elif env is EnvData2D:
        env_node = EnvData2D(dataset=dataset, limit_data=kwargs['limit_data'], window=kwargs.get('window', None), scaling=kwargs.get('scaling', 1), seed=seed)
    elif env is EnvPredictableNoise:
        env_node = EnvPredictableNoise(seed=seed)
    elif env is EnvRandom:
        env_node = EnvRandom(ndim=kwargs['ndim'], seed=seed)
    else:
        assert False
        
    (data_train, _, _), (data_test, _, _), _ = env_node.generate_training_data(n_train=n_train, 
                                                                               n_test=n_test, 
                                                                               n_validation=None, 
                                                                               actions=None, 
                                                                               noisy_dims=kwargs.get('noisy_dims', 0), 
                                                                               pca=kwargs.get('pca'), 
                                                                               pca_after_expansion=kwargs.get('pca_after_expansion'), 
                                                                               expansion=kwargs.get('expansion', 1),
                                                                               additive_noise=kwargs.get('additive_noise', 0), 
                                                                               whitening=kwargs.get('whitening', True))
    print data_train.shape, data_test.shape
    return [data_train, data_test], env_node


@echo
def train_model(algorithm, data_train, output_dim, seed, **kwargs):
    """
    A common interface to train different models.

    :param algorithm:
    :param data_train:
    :param output_dim:
    :param seed:
    :param kwargs:
    :return:
    """
    
    if algorithm == Algorithms.None or algorithm == Algorithms.PCA:
        return None
    elif algorithm == Algorithms.Random:
        return train_random(data_train=data_train, 
                    output_dim=output_dim, 
                    seed=seed,
                    seed2=kwargs.get('seed2',0))
                    #repetition_index=repetition_index)
    elif algorithm == Algorithms.SFA:
        return train_sfa(data_train=data_train, output_dim=output_dim)
    elif algorithm == Algorithms.SFFA:
        return train_sffa(data_train=data_train, output_dim=output_dim)
    elif algorithm == Algorithms.ForeCA:
        return train_foreca(data_train=data_train, output_dim=kwargs['output_dim_max'])
    elif algorithm == Algorithms.PFA:
        return train_pfa(data_train=data_train, 
                    output_dim=output_dim,
                    p=kwargs['p'],
                    K=kwargs['K'])
    elif algorithm == Algorithms.GPFA1:
        return train_gpfa(data_train=data_train,
                    k=kwargs['k'], 
                    p=kwargs.get('p', 1),
                    iterations=kwargs['iterations'], 
                    star_graph=False,
                    neighborhood_graph=kwargs.get('neighborhood_graph', False), 
                    weighted_edges=kwargs.get('weighted_edges', True),
                    causal_features=kwargs.get('causal_features', True),
                    generalized_eigen_problem=kwargs.get('generalized_eigen_problem', True),
                    output_dim=output_dim)
    elif algorithm == Algorithms.GPFA2:
        return train_gpfa(data_train=data_train, 
                    k=kwargs['k'], 
                    p=kwargs.get('p', 1),
                    iterations=kwargs['iterations'], 
                    star_graph=True,
                    neighborhood_graph=kwargs.get('neighborhood_graph', False), 
                    weighted_edges=kwargs.get('weighted_edges', True),
                    causal_features=kwargs.get('causal_features', True),
                    generalized_eigen_problem=kwargs.get('generalized_eigen_problem', True),
                    output_dim=output_dim)
    elif algorithm == Algorithms.HiRandom:
        return train_hi_random(data_train=data_train,
                            image_shape=(50, 50),
                            output_dim=output_dim,
                            expansion=kwargs.get('expansion', 2),
                            channels_xy_1=10,
                            spacing_xy_1=10,
                            channels_xy_n=2,
                            spacing_xy_n=1,
                            node_output_dim=10)
    elif algorithm == Algorithms.HiSFA:
        return train_hi_sfa(data_train=data_train, 
                            image_shape=(50,50), 
                            output_dim=output_dim, 
                            expansion=kwargs.get('expansion', 2),
                            channels_xy_1=10, 
                            spacing_xy_1=10, 
                            channels_xy_n=2, 
                            spacing_xy_n=1, 
                            node_output_dim=10)
    elif algorithm == Algorithms.HiSFFA:
        return train_hi_sffa(data_train=data_train, 
                             image_shape=(50,50), 
                             output_dim=output_dim, 
                             expansion=kwargs.get('expansion', 2),
                             channels_xy_1=10, 
                             spacing_xy_1=10, 
                             channels_xy_n=2, 
                             spacing_xy_n=1, 
                             node_output_dim=10)
    elif algorithm == Algorithms.HiPFA:
        return train_hi_pfa(data_train=data_train,
                            p=kwargs['p'],
                            K=kwargs['K'],
                            image_shape=(50,50), 
                            output_dim=output_dim, 
                            expansion=kwargs.get('expansion', 2),
                            channels_xy_1=10, 
                            spacing_xy_1=10, 
                            channels_xy_n=2, 
                            spacing_xy_n=1, 
                            node_output_dim=10)
    elif algorithm == Algorithms.HiGPFA:
        return train_hi_gpfa(data_train=data_train,
                             p=kwargs['p'],
                             k=kwargs['k'],
                             iterations=kwargs['iterations'],
                             image_shape=(50,50), 
                             output_dim=output_dim, 
                             expansion=kwargs.get('expansion', 2),
                             channels_xy_1=10, 
                             spacing_xy_1=10, 
                             channels_xy_n=2, 
                             spacing_xy_n=1, 
                             node_output_dim=10)
    else:
        assert False


#@mem.cache
def train_random(data_train, output_dim, seed, seed2):#, repetition_index):
    # rev: 2
    #fargs = update_seed_argument(output_dim=output_dim, repetition_index=repetition_index, seed=seed)
    model = random_node.RandomProjection(output_dim=output_dim, seed=seed+seed2)
    model.train(data_train)
    return model


#@mem.cache
def train_sfa(data_train, output_dim):
    model = mdp.nodes.SFANode(output_dim=output_dim)
    model.train(data_train)
    return model

 
#@mem.cache
def train_sffa(data_train, output_dim):
    model = sffa.SFFA(output_dim=output_dim)
    model.train(data_train)
    return model

 
@mem.cache
def train_foreca(data_train, output_dim):
    model = foreca_node.ForeCA(output_dim=output_dim, whitening=False)
    model.train(data_train)
    return model

 
#@mem.cache
def train_pfa(data_train, p, K, output_dim):
    model = PFANodeMDP.PFANode(p=p, k=K, affine=False, output_dim=output_dim)
    model.train(data_train)
    model.stop_training()
    return model

 
@mem.cache
def train_gpfa(data_train, k, iterations, star_graph, neighborhood_graph=False,
               weighted_edges=True, causal_features=True, generalized_eigen_problem=True,
               p=1, output_dim=1):
    model = gpfa_node.GPFA(k=k,
                      p=p,
                      output_dim=output_dim, 
                      iterations=iterations, 
                      star_graph=star_graph,
                      neighborhood_graph=neighborhood_graph,
                      weighted_edges=weighted_edges,
                      causal_features=causal_features,
                      generalized_eigen_problem=generalized_eigen_problem)
    model.train(data_train, is_whitened=True)
    return model


@mem.cache
def train_hi_sfa(data_train, image_shape, output_dim, expansion, channels_xy_1, 
                 spacing_xy_1, channels_xy_n, spacing_xy_n, node_output_dim):
    # rev: 10
    flow = build_hierarchy_flow(image_x=image_shape[1], 
                                image_y=image_shape[0], 
                                output_dim=output_dim, 
                                node_class=mdp.nodes.SFANode, 
                                node_output_dim=node_output_dim,
                                expansion=expansion,
                                channels_xy_1=channels_xy_1,
                                spacing_xy_1=spacing_xy_1,
                                channels_xy_n=channels_xy_n,
                                spacing_xy_n=spacing_xy_n,
                                node_kwargs={})
    flow.train(data_train)
    return flow


@mem.cache
def train_hi_sffa(data_train, image_shape, output_dim, expansion, channels_xy_1, 
                  spacing_xy_1, channels_xy_n, spacing_xy_n, node_output_dim):
    # rev: 0
    flow = build_hierarchy_flow(image_x=image_shape[1], 
                                image_y=image_shape[0], 
                                output_dim=output_dim, 
                                node_class=sffa.SFFA, 
                                node_output_dim=node_output_dim,
                                expansion=expansion,
                                channels_xy_1=channels_xy_1,
                                spacing_xy_1=spacing_xy_1,
                                channels_xy_n=channels_xy_n,
                                spacing_xy_n=spacing_xy_n,
                                node_kwargs={})
    flow.train(data_train)
    return flow


#@mem.cache
def train_hi_pfa(data_train, p, K, image_shape, output_dim, expansion, channels_xy_1, 
                 spacing_xy_1, channels_xy_n, spacing_xy_n, node_output_dim):
    # rev: 10
    flow = build_hierarchy_flow(image_x=image_shape[1], 
                                image_y=image_shape[0], 
                                output_dim=output_dim, 
                                node_class=PFANodeMDP.PFANode, 
                                node_output_dim=node_output_dim,
                                expansion=expansion,
                                channels_xy_1=channels_xy_1,
                                spacing_xy_1=spacing_xy_1,
                                channels_xy_n=channels_xy_n,
                                spacing_xy_n=spacing_xy_n,
                                node_kwargs={'p': p, 'k': K, 'affine': False})
    flow.train(data_train)
    return flow


@mem.cache
def train_hi_gpfa(data_train, p, k, iterations, image_shape, output_dim, expansion, 
                  channels_xy_1, spacing_xy_1, channels_xy_n, spacing_xy_n, node_output_dim):
    flow = build_hierarchy_flow(image_x=image_shape[1], 
                                image_y=image_shape[0], 
                                output_dim=output_dim, 
                                node_class=gpfa_node.GPFA,
                                node_output_dim=node_output_dim,
                                expansion=expansion,
                                channels_xy_1=channels_xy_1,
                                spacing_xy_1=spacing_xy_1,
                                channels_xy_n=channels_xy_n,
                                spacing_xy_n=spacing_xy_n,
                                node_kwargs={'k': k,
                                             'p': p,
                                             'iterations': iterations, 
                                             'star_graph': True,
                                             'neighborhood_graph': False,
                                             'weighted_edges': True,
                                             'causal_features': True,
                                             'generalized_eigen_problem': True})
    flow.train(data_train)
    return flow # rev 7


@mem.cache
def train_hi_random(data_train, image_shape, output_dim, expansion, channels_xy_1,
                 spacing_xy_1, channels_xy_n, spacing_xy_n, node_output_dim):
    # rev: 0
    flow = build_hierarchy_flow(image_x=image_shape[1],
                                image_y=image_shape[0],
                                output_dim=output_dim,
                                node_class=random_node.RandomProjection,
                                node_output_dim=node_output_dim,
                                expansion=expansion,
                                channels_xy_1=channels_xy_1,
                                spacing_xy_1=spacing_xy_1,
                                channels_xy_n=channels_xy_n,
                                spacing_xy_n=spacing_xy_n,
                                node_kwargs={})
    flow.train(data_train)
    return flow


def build_hierarchy_flow(image_x, image_y, output_dim, node_class, node_output_dim,
                         channels_xy_1, spacing_xy_1, channels_xy_n, spacing_xy_n, 
                         node_kwargs, expansion=1):

    switchboards = []
    layers = []
    
    while len(layers) == 0 or switchboards[-1].out_channels_xy[0] >= 2 or switchboards[-1].out_channels_xy[1] >= 2:

        first_layer = True if len(layers) == 0 else False
        last_layer  = True if len(layers) != 0 and switchboards[-1].out_channels_xy[0] <= 2 and switchboards[-1].out_channels_xy[1] <= 2 else False #layers[-1].output_dim <= 90 else False

        #if channels_xy_n == (2,1) or channels_xy_n == (1,2):
        #    channels_xy_n = (channels_xy_n[1], channels_xy_n[0])
        #    spacing_xy_n = (spacing_xy_n[1], spacing_xy_n[0])

        if first_layer:
            # first layer
            switchboards.append(mdp.hinet.Rectangular2dSwitchboard(in_channels_xy    = (image_x, image_y),
                                                                   field_channels_xy = channels_xy_1,
                                                                   field_spacing_xy  = spacing_xy_1,
                                                                   in_channel_dim    = 1,
                                                                   ignore_cover      = False))
        else:
            switchboards.append(mdp.hinet.Rectangular2dSwitchboard(in_channels_xy    = switchboards[-1].out_channels_xy,
                                                                   field_channels_xy = channels_xy_n,
                                                                   field_spacing_xy  = spacing_xy_n,
                                                                   in_channel_dim    = layers[-1][-1].output_dim,
                                                                   ignore_cover      = False))
    
        flow_nodes = []
        print 'creating layer with %s = %d nodes (last: %s)' % (switchboards[-1].out_channels_xy, switchboards[-1].output_channels, last_layer)
        for i in range(switchboards[-1].output_channels):
            nodes = []
            #nodes.append(mdp.nodes.IdentityNode(input_dim=switchboards[-1].out_channel_dim))
            #if not last_layer:
            nodes.append(mdp.nodes.NoiseNode(noise_args=(0, 1e-4), input_dim=switchboards[-1].out_channel_dim, dtype=np.float64))
            nodes.append(mdp.nodes.WhiteningNode(input_dim=nodes[-1].output_dim, output_dim=nodes[-1].output_dim, reduce=False, dtype=np.float64))
            nodes.append(node_class(input_dim=nodes[-1].output_dim, output_dim=node_output_dim, dtype=np.float64, **node_kwargs))
            #if not last_layer:
            if True:
                if expansion != 1:
                    nodes.append(mdp.nodes.GeneralExpansionNode(funcs=[f_identity, f_exp08], input_dim=nodes[-1].output_dim, dtype=np.float64)) #nodes.append(mdp.nodes.PolynomialExpansionNode(degree=expansion, input_dim=nodes[-1].output_dim, dtype=np.float64))
                    #nodes.append(mdp.nodes.NoiseNode(noise_args=(0, 1e-4), input_dim=nodes[-1].output_dim, dtype=np.float64))
                    nodes.append(mdp.nodes.WhiteningNode(input_dim=nodes[-1].output_dim, output_dim=nodes[-1].output_dim, reduce=False, dtype=np.float64))
                    nodes.append(node_class(input_dim=nodes[-1].output_dim, output_dim=output_dim if last_layer else node_output_dim, dtype=np.float64, **node_kwargs))
                nodes.append(mdp.nodes.CutoffNode(lower_bound=-4, upper_bound=4, input_dim=nodes[-1].output_dim, dtype=np.float64) if not last_layer else mdp.nodes.IdentityNode(input_dim=nodes[-1].output_dim, dtype=np.float64))
            flow_node = mdp.hinet.FlowNode(mdp.Flow(nodes, verbose=True))
            flow_nodes.append(flow_node)
            if i==0:
                for node in nodes:
                    print '%s: %d -> %d' % (node.__class__.__name__, node.input_dim, node.output_dim)
        layers.append(mdp.hinet.Layer(flow_nodes))
        
    hierarchy = []
    for switch, layer in zip(switchboards, layers):
        hierarchy.append(switch)
        hierarchy.append(layer)
        
    flow = mdp.Flow(hierarchy, verbose=True)
    
    print ''
    for node in flow:
        print node.__class__.__name__, node.input_dim, ' -> ', node.output_dim
        
    return flow



# @echo
# def calc_projected_data(env, dataset, algorithm, output_dim, n_train, n_test, #repetition_index,
#                         seed, noisy_dims=0, use_test_set=True, **kwargs):
#
#     [data_train, data_test], [pca1, pca2] = generate_training_data(env=env,
#                                                      dataset=dataset,
#                                                      n_train=n_train,
#                                                      n_test=n_test,
#                                                      noisy_dims=noisy_dims,
#                                                      #repetition_index=repetition_index,
#                                                      seed=seed,
#                                                      **kwargs)
#     print '%s: %d dimensions\n' % (dataset, data_train.shape[1])
#
#     model = train_model(algorithm=algorithm,
#                         data_train=data_train,
#                         output_dim=output_dim,
#                         seed=seed,
#                         #repetition_index=repetition_index,
#                         **kwargs)
#
#     if model is None:
#         if use_test_set:
#             projected_data = np.array(data_test, copy=True)
#         else:
#             projected_data = np.array(data_train, copy=True)
#     else:
#         if use_test_set:
#             projected_data = model.execute(data_test)
#         else:
#             projected_data = model.execute(data_train)
#         # reduce dim because ForeCA calculated output_dim_max dimensions
#         if  algorithm == Algorithms.ForeCA:# or \
#             #algorithm == Algorithms.HiSFA:
#             #algorithm == Algorithms.SFA or \
#             #algorithm == Algorithms.SFFA or \
#             projected_data = projected_data[:,:output_dim]
#
#     return projected_data, model, [data_train, data_test]



def dimensions_of_data(measure, dataset, algorithm, output_dim, n_train, n_test,
                       use_test_set, repetition_index, seed=None, **kwargs):
    
    _, _, data_chunks, _ = calc_projected_data(dataset=dataset,
                                            algorithm=algorithm, 
                                            output_dim=output_dim, 
                                            n_train=n_train,
                                            n_test=n_test, 
                                            use_test_set=use_test_set, 
                                            repetition_index=repetition_index, 
                                            seed=seed, **kwargs)
    
    return data_chunks[0].shape[1]



# @echo
# def prediction_error(measure, env, dataset, algorithm, output_dim, n_train, n_test,
#                      use_test_set, seed, **kwargs):
#     # rev: 11
#     projected_data, model, [data_train, data_test], _ = calc_projected_data(env=env,
#                                                                          dataset=dataset,
#                                                                          algorithm=algorithm,
#                                                                          output_dim=output_dim,
#                                                                          n_train=n_train,
#                                                                          n_test=n_test,
#                                                                          use_test_set=use_test_set,
#                                                                          #repetition_index=repetition_index,
#                                                                          seed=seed, **kwargs)
#
#     kwargs.update({'env': env,
#                    'dataset': dataset,
#                    'algorithm': algorithm,
#                    'output_dim': output_dim,
#                    'n_train': n_train,
#                    'n_test': n_test,
#                    'use_test_set': use_test_set,
#                    #'repetition_index': repetition_index,
#                    'seed': seed})
#     error = prediction_error_on_data(data=projected_data, measure=measure, model=model,
#                                      data_chunks=[data_train, data_test], **kwargs)
#     assert np.isfinite(error)
#     return error


# @echo
# def prediction_error_on_data(data, measure, model=None, data_chunks=None, **kwargs):
#
#     if data.ndim == 1:
#         n = data.shape[0]
#         data = np.array(data, ndmin=2).T
#         assert data.shape == (n,1)
#
#     if measure == Measures.delta:
#         return calc_delta(data=data, ndim=False)
#     elif measure == Measures.delta_ndim:
#         return calc_delta(data=data, ndim=True)
#     elif measure == Measures.omega:
#         return calc_omega(data=data, omega_dim=kwargs['output_dim']-1)
#     elif measure == Measures.omega_ndim:
#         return calc_omega_ndim(data=data)
#     elif measure == Measures.pfa:
#         return calc_autoregressive_error(data=data, model=model, p=kwargs['p'], data_train=data_chunks[0])
# #     elif measure == Measures.pfa_ndim:
# #         return calc_autoregressive_error_ndim(data=data,
# #                                          p=kwargs['p'],
# #                                          K=kwargs['K'],
# #                                          model=model,
# #                                          data_chunks=data_chunks)
#     elif measure == Measures.gpfa:
#         return calc_predictability_trace_of_avg_cov(x=data,
#                                                     k=kwargs.get('k_eval', kwargs['k']),
#                                                     p=kwargs['p'],
#                                                     ndim=False)
#     elif measure == Measures.gpfa_ndim:
#         return calc_predictability_trace_of_avg_cov(x=data,
#                                                     k=kwargs.get('k_eval', kwargs['k']),
#                                                     p=kwargs['p'],
#                                                     ndim=True)
#     elif measure == Measures.ndims:
#         return data_chunks[0].shape[1]
#     elif measure == Measures.angle_to_sfa_signals:
#         return calc_angle_to_sfa_signals(data, **kwargs)
#     elif measure == Measures.angle_to_p1:
#         return calc_angle_to_p1(data, **kwargs)
#     else:
#         assert False



@mem.cache
def calc_delta(data, ndim=False):
    assert data.ndim == 2
    sfa = mdp.nodes.SFANode()
    sfa.train(data)
    sfa.stop_training()
    if ndim:
        return sfa.d
    return np.sum(sfa.d)


#@mem.cache
@echo
def calc_autoregressive_error(data, model, p, data_train):
    if isinstance(model, PFANodeMDP.PFANode):
        W = model.W 
    else:
        # for instance when evaluating SFA signals with PFA measure
        # then PFA model needs to be trained first
        if model is not None:
            projected_data_train = model.execute(data_train)
        else:
            projected_data_train = data_train
        W = PFACoreUtil.calcRegressionCoeffRefImp(data=projected_data_train, p=p)
    return PFACoreUtil.empiricalRawErrorRefImp(data=data, W=W)


@mem.cache
def calc_predictability_trace_of_avg_cov(x, k, p, ndim):
    return gpfa_node.calc_predictability_trace_of_avg_cov( x=x, k=k, p=p, ndim=ndim )


@mem.cache
def calc_omega(data, omega_dim):
    from foreca.foreca_omega import omega
    return omega(data)[omega_dim]


@mem.cache
def calc_omega_ndim(data):
    from foreca.foreca_omega import omega
    return omega(data)


@mem.cache
def calc_angle_to_sfa_signals(data, **kwargs):
    kwargs['algorithm'] = Algorithms.HiSFA if kwargs.get('angle_to_hisfa', False) else Algorithms.SFA
    signals_sfa, _, _, _ = calc_projected_data(**kwargs)
    return principal_angles(signals_sfa, data)[0 if kwargs.get('min_principal_angle') else -1]  


@echo
#@mem.cache
def calc_angle_to_p1(data, **kwargs):
    #kwargs['seed'] = kwargs['seed'] + (hash(str(kwargs.get('p', 0))) % 1000000000)
    kwargs['seed2'] = 100000 #(hash(str(kwargs.get('p', 0))) % 1000000000)
    kwargs['p'] = 1
    signals, _, _, _ = calc_projected_data(**kwargs)
    return principal_angles(signals, data)[0 if kwargs.get('min_principal_angle') else -1]


#@echo
@mem.cache
def calc_min_delta_components(data, output_dim):
    assert data.ndim == 2
    deltas = []
    for signal in data.T:
        assert signal.ndim == 1
        signal = np.array(signal, ndmin=2).T
        sfa_node = mdp.nodes.SFANode(output_dim=1)
        sfa_node.train(signal)
        sfa_node.stop_training()
        deltas.append(sfa_node.d[0])
    deltas.sort()
    return np.sum(deltas[:output_dim])


def get_dataset_name(env, ds, latex=False):

    result = 'FOO'

    if env is EnvData:
        if ds is env_data.Datasets.STFT1:
            result = 'AUD_STFT1'
        elif ds is env_data.Datasets.STFT2:
            result = 'AUD_STFT2'
        elif ds is env_data.Datasets.STFT3:
            result = 'AUD_STFT3'
        elif ds is env_data.Datasets.Spectro1:
            result = 'AUD_SPECTRO1'
        elif ds is env_data.Datasets.Spectro2:
            result = 'AUD_SPECTRO2'
        elif ds is env_data.Datasets.Spectro3:
            result = 'AUD_SPECTRO3'
        elif ds is env_data.Datasets.EEG:
            result = 'PHY_EEG_GAL'
        elif ds is env_data.Datasets.EEG2:
            result = 'PHY_EEG_BCI'
        elif ds is env_data.Datasets.PHYSIO_EHG:
            result = 'PHY_EHG'
        elif ds is env_data.Datasets.EIGHT_EMOTION:
            result = 'PHY_EIGHT_EMOTION'
        elif ds is env_data.Datasets.PHYSIO_MGH:
            result = 'PHY_MGH_MF'
        elif ds is env_data.Datasets.PHYSIO_MMG:
            result = 'PHY_MMG'
        elif ds is env_data.Datasets.PHYSIO_UCD:
            result = 'PHY_UCDDB'
        elif ds is env_data.Datasets.HAPT:
            result = 'MISC_SBHAR'
        elif ds is env_data.Datasets.FIN_EQU_FUNDS:
            result = 'MISC_EQUITY_FUNDS'
        else:
            assert False
    elif env is EnvData2D:
        if ds is env_data2d.Datasets.Mario:
            result = 'VIS_SUPER_MARIO'
        elif ds is env_data2d.Datasets.SpaceInvaders:
            result = 'VIS_SPACE_INVADERS'
        elif ds is env_data2d.Datasets.Traffic:
            result = 'VIS_URBAN1'
        elif ds is env_data2d.Datasets.GoProBike:
            result = 'VIS_GOPRO_BIKE'
        else:
            assert False
    elif env is EnvRandom:
        result = 'MISC_NOISE'
    else:
        assert False

    if latex:
        result = result.replace('_', '\_')

    return result



if __name__ == '__main__':
    #set_cachedir(cachedir=None)
    #for measure in [Measures.delta, Measures.delta_ndim, Measures.gpfa, Measures.gpfa_ndim]:
    #    print prediction_error(measure=measure, 
    #                           dataset=Datasets.Mario_window, 
    #                           algorithm=Algorithms.GPFA2, 
    #                           output_dim=2, 
    #                           N=2000, 
    #                           k=10, 
    #                           p=1,
    #                           iterations=50,
    #                           seed=0)
    hierarchy = build_hierarchy_flow(image_x=50, 
                               image_y=50, 
                               output_dim=10, 
                               node_class=mdp.nodes.SFANode, 
                               node_output_dim=10, 
                               channels_xy_1=10, 
                               spacing_xy_1=10, 
                               channels_xy_n=2, 
                               spacing_xy_n=1, 
                               node_kwargs={}, 
                               expansion=2)
        
        
