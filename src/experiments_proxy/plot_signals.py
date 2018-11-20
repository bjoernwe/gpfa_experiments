import matplotlib.pyplot as plt
import numpy as np

from envs import env_data, env_data2d

import experiment_base as eb
import parameters



def main():

    plot_alg_names = {eb.Algorithms.None: 'input data',
                      eb.Algorithms.Random: 'Random',
                      eb.Algorithms.SFA: 'SFA',
                      eb.Algorithms.SFFA: "SFA'",
                      eb.Algorithms.ForeCA: 'ForeCA',
                      eb.Algorithms.PFA: 'PFA',
                      eb.Algorithms.GPFA2: 'GPFA',
                      }

    algs = [#eb.Algorithms.Random,
            eb.Algorithms.SFA,
            eb.Algorithms.ForeCA,
            #eb.Algorithms.SFFA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]

    dataset_list = [env_data.Datasets.STFT1,
                    #env_data.Datasets.STFT2,
                    env_data.Datasets.STFT3,
                    #env_data.Datasets.EEG2,
                    #env_data.Datasets.EEG,
                    #env_data.Datasets.EIGHT_EMOTION,
                    #env_data.Datasets.PHYSIO_EHG,
                    env_data.Datasets.PHYSIO_MGH,
                    #env_data.Datasets.PHYSIO_MMG,
                    env_data.Datasets.PHYSIO_UCD,
                    env_data2d.Datasets.GoProBike,
                    #env_data2d.Datasets.SpaceInvaders,
                    env_data2d.Datasets.Mario,
                    env_data2d.Datasets.Traffic,
                    #env_data.Datasets.FIN_EQU_FUNDS,
                    env_data.Datasets.HAPT,
                    ]

    results_test  = {}
    results_train = {}
    
    for alg in algs:
        #r = 1#parameters.algorithm_args[alg].get('repetitions', 50)
        seed = [1] #range(r)
        results_test[alg]  = parameters.get_signals(alg, dataset_list=dataset_list, overide_args={'seed': seed, 'output_dim': 1})
        results_train[alg] = parameters.get_signals(alg, dataset_list=dataset_list, overide_args={'seed': seed, 'output_dim': 1, 'use_test_set': False})
        
    alphas = np.linspace(0, 1, 6)[::-1]

    figsize = (20,11)
    plt.figure(figsize=figsize)

    idat = -1
    for _, dataset_args in enumerate(parameters.dataset_args):

        env = dataset_args['env']
        dataset = dataset_args['dataset']
        if not dataset in dataset_list:
            continue
        idat += 1

        for ialg, alg in enumerate(algs):

            if not dataset in results_train[alg]:
                continue

            signals_train = results_train[alg][dataset]['projected_data'][:,:,0] # only first iteration
            signals_test  = results_test[alg][dataset]['projected_data'][:,:,0]

            print (idat,ialg)
            plt.subplot2grid(shape=(len(dataset_list),4), loc=(idat,ialg))

            for i in range(1)[::-1]:
                signal_train = signals_train[:,i]
                signal_test  = signals_test[:,i]
                signal_train = signal_train[-500:]
                signal_test  = signal_test[:200]
                n_train = signal_train.shape[0]
                n_test  = signal_test.shape[0]
                sign = np.sign(np.correlate(signal_train, results_train[eb.Algorithms.SFA][dataset]['projected_data'][:,i,0])[0])
                plt.plot(range(n_train), sign*signal_train, c='b', alpha=alphas[i])
                #plt.plot(range(n_train, n_train+n_test), sign*signal_test, c='r', alpha=alphas[i])

            plt.yticks([])
            plt.xticks([])

            if idat == 0:
                plt.title(plot_alg_names[alg], fontsize=12)
            if ialg == 0:
                plt.ylabel(eb.get_dataset_name(env=env, ds=dataset), rotation=0, horizontalalignment='right', verticalalignment='top')

    plt.subplot2grid(shape=(len(dataset_list),4), loc=(0,1))
    plt.title('ForeCA', fontsize=12)
    plt.gca().axis('off')

    plt.subplots_adjust(left=0.1, right=.99, bottom=0.04, top=.96, wspace=.05)
    plt.savefig('fig_signals.eps')
    plt.show()



if __name__ == '__main__':
    main()
    