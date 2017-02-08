import joblib
import mdp
import numpy as np
import sys

from enum import Enum

import foreca.foreca_node as foreca_node
import gpfa
import sffa

#sys.path.append('/home/weghebvc/workspace/git/explot/src/')
#import explot as ep

sys.path.append('/home/weghebvc/workspace/git/GNUPFA/src/')
import PFANodeMDP

sys.path.append('/home/weghebvc/workspace/git/environments_new/src/')
#from envs.environment import Noise
from envs import env_data
from envs import env_data2d
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_kai import EnvKai
from envs.env_random import EnvRandom

sys.path.append('/home/weghebvc/workspace/git/GNUPFA')
import PFACoreUtil



# prepare joblib.Memory
default_cachedir = '/scratch/weghebvc'
mem = joblib.Memory(cachedir=default_cachedir, verbose=1)


#Datasets = Enum('Datasets', 'Random Crowd1 Crowd2 Crowd3 Dancing Mouth Face RatLab Kai Mario Mario_window Mario_window_8 EEG EEG2 EEG2_stft_128 MEG SpaceInvaders SpaceInvaders_window SpaceInvaders_window_8 Traffic Traffic_window Traffic_window_8 Tumor STFT1 STFT2 STFT3')

Algorithms = Enum('Algorithms', 'None Random SFA SFFA ForeCA PFA GPFA1 GPFA2')

Measures = Enum('Measures', 'delta delta_ndim omega omega_ndim pfa gpfa gpfa_ndim ndims angle_to_sfa')



def set_cachedir(cachedir=None):
    """
    Call this method to change the joblib caching of this module.
    """
    global mem
    mem = joblib.Memory(cachedir=cachedir, verbose=1)
    return



def update_seed_argument(remove_args=None, **kwargs):
    """
    Helper function that replaces the the seed argument by a new seed that
    depends on all arguments. If repetition_index is given it will be removed.
    """
    new_seed = hash(frozenset(kwargs.items())) % np.iinfo(np.uint32).max
    if 'repetition_index' in kwargs:
        kwargs.pop('repetition_index')
    if remove_args:
        for arg in remove_args:
            if arg in kwargs:
                kwargs.pop(arg)
    kwargs['seed'] = new_seed
    return kwargs



def generate_training_data(env, dataset, n_train, n_test, repetition_index, seed=None, **kwargs):

    if env is EnvData:
        fargs = update_seed_argument(remove_args=['n_train'], n_train=n_train, limit_data=kwargs['limit_data'], repetition_index=repetition_index, seed=seed)
        env_node = EnvData(dataset=dataset, **fargs)
    elif env is EnvData2D:
        fargs = update_seed_argument(remove_args=['n_train'], n_train=n_train, limit_data=kwargs['limit_data'], window=kwargs.get('window', None), scaling=1, repetition_index=repetition_index, seed=seed)
        env_node = EnvData2D(dataset=dataset, **fargs)
    elif env is EnvKai:
        fargs = update_seed_argument(remove_args=['n_train'], n_train=n_train, repetition_index=repetition_index, seed=seed) 
        env_node = EnvKai(**fargs)
    elif env is EnvRandom:
        fargs = update_seed_argument(remove_args=['n_train'], n_train=n_train, ndim=kwargs['ndim'], repetition_index=repetition_index, seed=seed) 
        env_node = EnvRandom(**fargs)
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
                                                                               whitening=True)
    return [data_train, data_test]



def train_model(algorithm, data_train, output_dim, seed, repetition_index, **kwargs):
    
    if algorithm == Algorithms.None:
        return None
    elif algorithm == Algorithms.Random:
        return train_random(data_train=data_train, 
                    output_dim=output_dim, 
                    seed=seed, 
                    repetition_index=repetition_index)
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
                    variance_graph=True,
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
                    variance_graph=False,
                    neighborhood_graph=kwargs.get('neighborhood_graph', False), 
                    weighted_edges=kwargs.get('weighted_edges', True),
                    causal_features=kwargs.get('causal_features', True),
                    generalized_eigen_problem=kwargs.get('generalized_eigen_problem', True),
                    output_dim=output_dim)
    else:
        assert False



#@mem.cache
def train_random(data_train, output_dim, seed, repetition_index):
    # rev: 2
    fargs = update_seed_argument(output_dim=output_dim, repetition_index=repetition_index, seed=seed)
    model = gpfa.RandomProjection(**fargs)
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
    model = foreca_node.ForeCA(output_dim=output_dim)
    model.train(data_train)
    return model
 
 
 
#@mem.cache
def train_pfa(data_train, p, K, output_dim):
    model = PFANodeMDP.PFANode(p=p, k=K, affine=False, output_dim=output_dim)
    model.train(data_train)
    model.stop_training()
    return model
 

 
@mem.cache
def train_gpfa(data_train, k, iterations, variance_graph, neighborhood_graph=False, 
               weighted_edges=True, causal_features=True, generalized_eigen_problem=True,
               p=1, output_dim=1):
    model = gpfa.gPFA(k=k, 
                      p=p,
                      output_dim=output_dim, 
                      iterations=iterations, 
                      variance_graph=variance_graph,
                      neighborhood_graph=neighborhood_graph,
                      weighted_edges=weighted_edges,
                      causal_features=causal_features,
                      generalized_eigen_problem=generalized_eigen_problem)
    model.train(data_train)
    return model



def calc_projected_data(env, dataset, algorithm, output_dim, n_train, n_test, repetition_index, 
                        noisy_dims=0, use_test_set=True, seed=None, **kwargs):
    
    [data_train, data_test] = generate_training_data(env=env,
                                                     dataset=dataset, 
                                                     n_train=n_train,
                                                     n_test=n_test, 
                                                     noisy_dims=noisy_dims,
                                                     repetition_index=repetition_index, 
                                                     seed=seed,
                                                     **kwargs)
    print '%s: %d dimensions\n' % (dataset, data_train.shape[1])
    
    model = train_model(algorithm=algorithm, 
                        data_train=data_train, 
                        output_dim=output_dim, 
                        seed=seed,
                        repetition_index=repetition_index,
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
        if algorithm == Algorithms.ForeCA:
            projected_data = projected_data[:,:output_dim]
        
    return projected_data, model, [data_train, data_test]



def dimensions_of_data(measure, dataset, algorithm, output_dim, n_train, n_test, 
                       use_test_set, repetition_index, seed=None, **kwargs):
    
    _, _, data_chunks = calc_projected_data(dataset=dataset, 
                                            algorithm=algorithm, 
                                            output_dim=output_dim, 
                                            n_train=n_train,
                                            n_test=n_test, 
                                            use_test_set=use_test_set, 
                                            repetition_index=repetition_index, 
                                            seed=seed, **kwargs)
    
    return data_chunks[0].shape[1]
    
    

def prediction_error(measure, env, dataset, algorithm, output_dim, n_train, n_test, use_test_set, 
                     repetition_index=None, seed=None, **kwargs):
    # rev: 4
    projected_data, model, [data_train, data_test] = calc_projected_data(env=env,
                                                                         dataset=dataset, 
                                                                         algorithm=algorithm, 
                                                                         output_dim=output_dim, 
                                                                         n_train=n_train,
                                                                         n_test=n_test, 
                                                                         use_test_set=use_test_set, 
                                                                         repetition_index=repetition_index, 
                                                                         seed=seed, **kwargs)
    
    kwargs.update({'env': env, 
                   'dataset': dataset, 
                   'algorithm': algorithm, 
                   'output_dim': output_dim, 
                   'n_train': n_train, 
                   'n_test': n_test, 
                   'use_test_set': use_test_set, 
                   'repetition_index': repetition_index, 
                   'seed': seed})
    return prediction_error_on_data(data=projected_data, measure=measure, model=model, 
                                    data_chunks=[data_train, data_test], **kwargs)
    
    

def prediction_error_on_data(data, measure, model=None, data_chunks=None, **kwargs):

    if data.ndim == 1:
        n = data.shape[0]
        data = np.array(data, ndmin=2).T
        assert data.shape == (n,1)
        
    if measure == Measures.delta:
        return calc_delta(data=data, ndim=False)
    elif measure == Measures.delta_ndim:
        return calc_delta(data=data, ndim=True)
    elif measure == Measures.omega:
        return calc_omega(data=data, omega_dim=kwargs['output_dim']-1)
    elif measure == Measures.omega_ndim:
        return calc_omega_ndim(data=data)
    elif measure == Measures.pfa:
        return calc_autoregressive_error(data=data, model=model, p=kwargs['p'], data_train=data_chunks[0])
#     elif measure == Measures.pfa_ndim:
#         return calc_autoregressive_error_ndim(data=data, 
#                                          p=kwargs['p'], 
#                                          K=kwargs['K'],
#                                          model=model,
#                                          data_chunks=data_chunks)
    elif measure == Measures.gpfa:
        return calc_predictability_trace_of_avg_cov(x=data, 
                                                    k=kwargs['k_eval'],#.get('k_eval', kwargs['k']), 
                                                    p=kwargs['p'],
                                                    ndim=False)
    elif measure == Measures.gpfa_ndim:
        return calc_predictability_trace_of_avg_cov(x=data, 
                                                    k=kwargs['k_eval'],#.get('k_eval', kwargs['k']), 
                                                    p=kwargs['p'],
                                                    ndim=True)
    elif measure == Measures.ndims:
        return data_chunks[0].shape[1]
    elif measure == Measures.angle_to_sfa:
        return calc_angle_to_sfa_signals(data, **kwargs)
    else:
        assert False
    
    
    
@mem.cache
def calc_delta(data, ndim=False):
    sfa = mdp.nodes.SFANode()
    sfa.train(data)
    sfa.stop_training()
    if ndim:
        return sfa.d
    return np.sum(sfa.d)



@mem.cache
def calc_autoregressive_error(data, model, p, data_train):
    if isinstance(model, PFANodeMDP.PFANode):
        W = model.W 
    else:
        # for instance when evaluating SFA signals with PFA measure
        # then PFA model needs to be trained first
        projected_data_train = model.execute(data_train)
        W = PFACoreUtil.calcRegressionCoeffRefImp(data=projected_data_train, p=p)
    return PFACoreUtil.empiricalRawErrorRefImp(data=data, W=W)



@mem.cache
def calc_predictability_trace_of_avg_cov(x, k, p, ndim):
    return gpfa.calc_predictability_trace_of_avg_cov(x=x, k=k, p=p, ndim=ndim)



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
    kwargs['algorithm'] = Algorithms.SFA
    signals_sfa, _, _ = calc_projected_data(**kwargs)
    return _principal_angle(signals_sfa, data) 



def _principal_angle(A, B):
    """A and B must be column-orthogonal.
    Golub: Matrix Computations, 1996
    [http://www.disi.unige.it/person/BassoC/teaching/python_class02.pdf]
    """
    #A = np.array(A, copy=True)
    #B = np.array(B, copy=True)
    if A.ndim == 1:
        A = np.array(A, ndmin=2).T
    if B.ndim == 1:
        B = np.array(B, ndmin=2).T
    assert A.ndim == B.ndim == 2
    A = np.linalg.qr(A)[0]
    B = np.linalg.qr(B)[0]
    _, S, _ = np.linalg.svd(np.dot(A.T, B))
    return np.arccos(min(S.min(), 1.0))



def get_dataset_name(env, ds, latex=False):
    
    result = 'FOO'
    
    if env is EnvData:
        if ds is env_data.Datasets.STFT1:
            result = 'AUD_STFT1'
        elif ds is env_data.Datasets.STFT2:
            result = 'AUD_STFT2'
        elif ds is env_data.Datasets.STFT3:
            result = 'AUD_STFT3'
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
    for measure in [Measures.delta, Measures.delta_ndim, Measures.gpfa, Measures.gpfa_ndim]:
        print prediction_error(measure=measure, 
                               dataset=Datasets.Mario_window, 
                               algorithm=Algorithms.GPFA2, 
                               output_dim=2, 
                               N=2000, 
                               k=10, 
                               p=1,
                               iterations=50,
                               seed=0)

