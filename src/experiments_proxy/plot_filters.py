import matplotlib
import matplotlib.pyplot as plt
import mdp
import mkl
import numpy as np

import foreca.foreca_node as foreca_node
import gpfa_node
import random_node
import PFANodeMDP

from envs import env_data
from envs import env_data2d

import experiment_base as eb
import parameters
import parameters_hi


def get_features_from_model(model):
    A = None
    W = None
    if isinstance(model, mdp.nodes.SFANode):
        A = model.sf
    elif isinstance(model, foreca_node.ForeCA):
        A = model.U  # Extraction w/o whitening, make sure whitening was off!
        W = model.W
    elif isinstance(model, PFANodeMDP.PFANode):
        A = model.Ar
    elif isinstance(model, gpfa_node.GPFA):
        A = model.U
        if model.whitening is not None:
            W = model.whitening.v
    elif isinstance(model, random_node.RandomProjection):
        A = model.U
    elif model is None: # PCA
        pass
    else:
        print type(model)
        assert False
    return A, W


def main():

    mkl.set_num_threads(1)

    plot_alg_names = {eb.Algorithms.None:   'input data',
                      eb.Algorithms.Random: 'Random',
                      eb.Algorithms.SFA:    'SFA',
                      eb.Algorithms.SFFA:   "SFA'",
                      eb.Algorithms.ForeCA: 'ForeCA',
                      eb.Algorithms.PFA:    'PFA',
                      eb.Algorithms.GPFA2:  'GPFA',
                      }
    
    algs = [#eb.Algorithms.None,
            eb.Algorithms.PCA,
            eb.Algorithms.Random,
            eb.Algorithms.SFA,
            #eb.Algorithms.ForeCA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]
    
    algs_hi = [#eb.Algorithms.HiSFA,
               #eb.Algorithms.HiPFA,
               #eb.Algorithms.HiGPFA,
               ]
    
    #repetitions = 3#parameters.default_args_explot['repetitions']
    #repetitions_hi = 3#parameters_hi.default_args_explot['repetitions']

    results_train = {}
    for alg in algs:
        stack_result = not alg is eb.Algorithms.None
        #r = 3#parameters.algorithm_args[alg].get('repetitions', repetitions)
        results_train[alg] = parameters.get_signals(alg,
                                                    overide_args={'use_test_set': False, 'output_dim': 5, 'seed': 1},#, 'n_train': 10000, 'n_test': 100},
                                                    stack_result=stack_result,
                                                    dataset_list=[#env_data.Datasets.STFT1,
                                                                  #env_data.Datasets.STFT2,
                                                                  #env_data.Datasets.STFT3,
                                                                  env_data2d.Datasets.GoProBike,
                                                                  env_data2d.Datasets.RatLab,
                                                                  env_data2d.Datasets.SpaceInvaders,
                                                                  env_data2d.Datasets.Mario,
                                                                  env_data2d.Datasets.Traffic
                                                                  ])
    #for alg in algs_hi:
    #    r = 3#parameters_hi.algorithm_args[alg].get('repetitions', repetitions_hi)
    #    results_train[alg] = parameters_hi.get_signals(alg, overide_args={'use_test_set': False, 'output_dim': 5, 'seed': range(1,r+1)})

    #plt.plot(results_train[alg][env_data2d.Datasets.RatLab]['projected_data'])
    #plt.show()

    for alg in algs:

        plt.figure()

        id = -1
        for dataset_args in parameters.dataset_args:

            env = dataset_args['env']
            dataset = dataset_args['dataset']
            if not dataset in results_train[alg]:
                continue
            if not dataset in [env_data.Datasets.STFT1,
                               env_data.Datasets.STFT2,
                               env_data.Datasets.STFT3,
                               env_data2d.Datasets.RatLab,
                               env_data2d.Datasets.GoProBike,
                               env_data2d.Datasets.SpaceInvaders,
                               env_data2d.Datasets.Mario,
                               env_data2d.Datasets.Traffic]:
                continue
            id += 1

            print dataset
            data_train = results_train[alg][dataset]['data_train']
            model = results_train[alg][dataset]['model']
            env_node = results_train[alg][dataset]['env_node']
            pca_node = env_node.pca1
            whitening = env_node.whitening_node
            V = pca_node.v
            W = whitening.v
            Winv = np.linalg.inv(whitening.v)
            U, W2 = get_features_from_model(model)
            assert W2 is None

            # plot PCA matrix
            #plt.subplot(1, 3, id+1)
            #plt.imshow(V, cmap=plt.get_cmap('Greys'), interpolation='none')
            #continue

            # plot whitening matrix
            #print np.diag(W)
            #plt.subplot(1, 3, id+1)
            #plt.imshow(W, cmap=plt.get_cmap('Greys'), interpolation='none')
            #continue

            #dat1 = data_train.dot(Winv)
            #dat2 = dat1.dot(V.T)
            #print np.cov(dat1.T)
            #print np.cov(dat1.T).shape
            #print np.cov(dat2.T)
            #print np.cov(dat2.T).shape

            #plt.figure()
            #plt.imshow(np.cov(data_train.T), cmap=plt.get_cmap('Greys'))
            #plt.figure()
            #plt.imshow(np.cov(dat1.T), cmap=plt.get_cmap('Greys'))
            #plt.subplot(1, 3, id+1)
            #plt.imshow(np.cov(dat2.T), cmap=plt.get_cmap('Greys'), interpolation='none')
            #continue

            if alg is eb.Algorithms.PCA:
                assert U is None
                #F = V.dot(Winv)
                #F = V.dot(W)
                F = V
            else:
                #F = V.dot(Winv.dot(U))
                F = V.dot(W.dot(U))
            assert F.shape == (400, 5)
            #assert F.shape == (512, 5)
            #F = V
            #print F.shape

            # normalization
            #print np.sign(np.mean(np.sign(F), axis=0))
            F *= -np.sign(np.mean(F, axis=0))
            #F *= -np.sign(F[190,:])
            #print F

            #plt.figure()
            for iv in range(5):
                plt.subplot2grid(shape=(4,5), loc=(id,iv))
                filter = F[:,iv].reshape((20,20))
                plt.imshow(filter, cmap=plt.get_cmap('Greys'), interpolation='none')#, vmin=-1e-3, vmax=1e-3)
                #plt.bar(range(512), filter)

    #plt.subplots_adjust(left=0.1, right=.99, bottom=0.04, top=.96, wspace=.05)
    #plt.savefig('fig_spectra.eps')
    plt.show()


if __name__ == '__main__':
    main()
    
