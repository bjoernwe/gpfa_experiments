import matplotlib.pyplot as plt
import mkl
import numpy as np

from envs.env_random import EnvRandom

#import experiments_proxy.experiment_base_proxy as eb
import experiment_base as eb
import parameters



def main():

    mkl.set_num_threads(1)

    use_test_set = False

    plot_alg_names = {eb.Algorithms.Random: 'Random',
                      eb.Algorithms.SFA:    'SFA',
                      eb.Algorithms.SFFA:   "SFA'",
                      eb.Algorithms.ForeCA: 'ForeCA',
                      eb.Algorithms.PFA:    'PFA',
                      eb.Algorithms.GPFA2:  'GPFA',
                      }

    algs = [#eb.Algorithms.Random,
            eb.Algorithms.ForeCA,
            #eb.Algorithms.SFA,
            #eb.Algorithms.SFFA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]

    results_random = {}
    for min_principal_angle in [False, True]:
        results_random[min_principal_angle] = parameters.get_results(eb.Algorithms.Random,
                                                                     overide_args={'measure': eb.Measures.angle_to_sfa_signals,
                                                                                   'min_principal_angle': min_principal_angle,
                                                                                   'use_test_set': use_test_set})

    results_angle = {}
    for alg in algs:
        results_angle[alg] = {}
        for min_principal_angle in [False, True]:
            results_angle[alg][min_principal_angle] = parameters.get_results(alg,
                                                                             overide_args={
                                                                                 'measure': eb.Measures.angle_to_sfa_signals,
                                                                                 'min_principal_angle': min_principal_angle,
                                                                                 'use_test_set': use_test_set})

    plt.figure(figsize=(10, 2.1))
    plt.suptitle(eb.get_dataset_name(env=EnvRandom, ds=None, latex=False))

    for i, alg in enumerate(algs):

        # subplots
        plt.subplot(1, 3, i + 1)

        for min_principal_angle in [False, True]:

            # random
            values_random = results_random[min_principal_angle][None].values * ( 180. / np.pi)
            d, _ = values_random.shape
            plt.errorbar(x=range(1,d+1), y=np.mean(values_random, axis=1), yerr=np.std(values_random, axis=1), color='silver', ls='--', dashes=(5,2), zorder=0)

        for min_principal_angle in [False, True]:

            # angles
            values = results_angle[alg][min_principal_angle][None].values * ( 180. / np.pi)
            d, _ = values.shape
            plt.errorbar(x=range(1,d+1), y=np.mean(values, axis=1), yerr=np.std(values, axis=1), color='green' if min_principal_angle else 'blue', zorder=10)
            xlim_max = 5.5 #if alg is eb.Algorithms.ForeCA else 10.5
            plt.xlim(.5, xlim_max)
            #plt.ylim(-.2, np.pi/2+.2)
            plt.ylim(-5, 99)
            if i == 0:
                plt.ylabel('angle [deg]')
            else:
                plt.gca().set_yticklabels([])
            plt.xlabel('M')

        # title
        plt.title(plot_alg_names[alg], fontsize=12)
        plt.subplots_adjust(hspace=.4, wspace=.15, left=0.17, right=.86, bottom=.25, top=.79)

    plt.savefig('fig_angles_random.eps')
    plt.show()



if __name__ == '__main__':
    main()
    
