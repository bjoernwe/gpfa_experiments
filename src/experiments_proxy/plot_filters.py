import matplotlib
import matplotlib.pyplot as plt
import mdp
import mkl
import numpy as np

import foreca.foreca_node as foreca_node
import gpfa_node
import random_node
import PFANodeMDP

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
            eb.Algorithms.Random,
            eb.Algorithms.SFA,
            eb.Algorithms.ForeCA,
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
                                                    overide_args={'use_test_set': False, 'output_dim': 5, 'seed': 0}, #, 'n_train': 18000, 'n_test': 500},
                                                    stack_result=stack_result,
                                                    dataset_list=[env_data2d.Datasets.SpaceInvaders, env_data2d.Datasets.Mario, env_data2d.Datasets.Traffic])
    #for alg in algs_hi:
    #    r = 3#parameters_hi.algorithm_args[alg].get('repetitions', repetitions_hi)
    #    results_train[alg] = parameters_hi.get_signals(alg, overide_args={'use_test_set': False, 'output_dim': 5, 'seed': range(1,r+1)})

    for alg in algs:

        plt.figure()

        id = -1
        for dataset_args in parameters.dataset_args:

            env = dataset_args['env']
            dataset = dataset_args['dataset']
            if not dataset in results_train[alg]:
                continue
            if not dataset in [env_data2d.Datasets.SpaceInvaders, env_data2d.Datasets.Mario, env_data2d.Datasets.Traffic]:
                continue
            id += 1

            print dataset
            data_train = results_train[alg][dataset]['data_train']
            model = results_train[alg][dataset]['model']
            env_node = results_train[alg][dataset]['env_node']
            pca_node = env_node.pca1
            whitening = env_node.whitening_node
            V = pca_node.v
            Winv = np.linalg.inv(whitening.v)
            U, W2 = get_features_from_model(model)
            assert W2 is None

            dat1 = data_train.dot(Winv)
            dat2 = dat1.dot(V.T)
            #print np.cov(dat1.T)
            #print np.cov(dat1.T).shape
            #print np.cov(dat2.T)
            #print np.cov(dat2.T).shape

            #plt.figure()
            #plt.imshow(np.cov(data_train.T), cmap=plt.get_cmap('Greys'))
            #plt.figure()
            #plt.imshow(np.cov(dat1.T), cmap=plt.get_cmap('Greys'))
            #plt.figure()
            #plt.imshow(np.cov(dat2.T), cmap=plt.get_cmap('Greys'))

            F = V.dot(Winv.dot(U))
            assert F.shape == (400, 5)

            # normalization
            print np.sign(np.mean(np.sign(F), axis=0))
            F *= -np.sign(np.mean(F, axis=0))
            #F *= -np.sign(F[190,:])

            #plt.figure()
            for iv in range(5):
                plt.subplot2grid(shape=(3,5), loc=(id,iv))
                plt.imshow(F[:,iv].reshape((20,20)), cmap=plt.get_cmap('Greys'))

    # figsize = (20,11)
    # plt.figure(figsize=figsize)
    #
    # for idat, dataset_args in enumerate(parameters.dataset_args):
    #
    #     for ialg, alg in enumerate(algs + algs_hi):
    #
    #         env = dataset_args['env']
    #         dataset = dataset_args['dataset']
    #         if not dataset in results_train[alg]:
    #             continue
    #
    #         spectra_list = []
    #         if alg is eb.Algorithms.None:
    #             for input_data in results_train[alg][dataset]['projected_data']:
    #                 for signal in input_data.T:
    #                     spectrum = np.fft.fft(signal)
    #                     spectra_list.append(spectrum)
    #         else:
    #             for signal in results_train[alg][dataset]['projected_data'][:,0,:].T:
    #                 spectrum = np.fft.fft(signal)
    #                 spectra_list.append(spectrum)
    #         spectra = np.vstack(spectra_list).T
    #         spectrum = np.mean(spectra, axis=-1)    # average over repetitions and dimensions
    #
    #         signal_length = spectrum.shape[0]
    #         power_spectrum = np.abs(spectrum)[:signal_length//2]
    #
    #         plt.subplot2grid(shape=(16,8), loc=(idat,ialg))
    #         plt.plot(power_spectrum, c='b')
    #         plt.xticks([])
    #         plt.yticks([])
    #         margin = signal_length // 60
    #         plt.xlim([-margin, signal_length//2 + margin])
    #         if idat == 0:
    #             plt.title(plot_alg_names[alg], fontsize=12)
    #         elif idat == 15:
    #             plt.xlabel('frequencies')
    #         elif idat == 12 and ialg >= 5:
    #             plt.xlabel('frequencies')
    #         if ialg == 0:
    #             plt.ylabel(eb.get_dataset_name(env=env, ds=dataset), rotation=0, horizontalalignment='right', verticalalignment='top')
        
    #plt.subplots_adjust(left=0.1, right=.99, bottom=0.04, top=.96, wspace=.05)
    #plt.savefig('fig_spectra.eps')
    plt.show()


if __name__ == '__main__':
    main()
    
