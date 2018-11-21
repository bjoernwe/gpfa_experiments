import matplotlib.pyplot as plt
import mkl
import numpy as np

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

    algs = [eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]

    search_range_p = {eb.Algorithms.PFA:   [1,2,4,6,8,10],
                      eb.Algorithms.GPFA2: [1,2,4,6]}

    results_random = {}
    for min_principal_angle in [False, True]:
        results_random[min_principal_angle] = parameters.get_results(eb.Algorithms.Random,
                                                                     overide_args={'measure': eb.Measures.angle_to_p1,
                                                                                   'min_principal_angle': min_principal_angle,
                                                                                   'use_test_set': use_test_set,
										   'output_dim': 5,
						                                   'output_dim_max': 5,
										   'p': search_range_p[alg]})

    results_angle = {}
    for alg in algs:
        results_angle[alg] = {}
        for min_principal_angle in [False, True]:
            override_args = {'measure': eb.Measures.angle_to_p1,
                             'min_principal_angle': min_principal_angle,
                             'use_test_set': use_test_set,
                             'output_dim': 5,
                             'output_dim_max': 5,
                             'p': search_range_p[alg]}
            results_angle[alg][min_principal_angle]  = parameters.get_results(alg, overide_args=override_args)
        
    for _, alg in enumerate(algs):
        
        figsize = (10,3.2) if alg is eb.Algorithms.ForeCA else (10,6)
        plt.figure(figsize=figsize)
        plt.suptitle(plot_alg_names[alg])
            
        idx = 0
        for _, dataset_args in enumerate(parameters.dataset_args):

            env = dataset_args['env']
            dataset = dataset_args['dataset']
            if not dataset in results_angle[alg][False] or dataset is None:
                continue

            # subplots
            n_rows = 2 if alg is eb.Algorithms.ForeCA else 4
            plt.subplot(n_rows, 4, idx + 1)
            # plt.subplot2grid(shape=(n_algs,4), loc=(a,3))

            for min_principal_angle in [False, True]:

                # random
                values_random = results_random[min_principal_angle][dataset].values * ( 180. / np.pi)
                plt.errorbar(x=search_range_p[alg], y=np.mean(values_random, axis=1), yerr=np.std(values_random, axis=1), color='silver', ls='--', dashes=(5,2), zorder=0)

            for min_principal_angle in [False, True]:

                # angles
                values = results_angle[alg][min_principal_angle][dataset].values * ( 180. / np.pi)
                d, _ = values.shape
                plt.errorbar(x=search_range_p[alg], y=np.mean(values, axis=1), yerr=np.std(values, axis=1), color='green' if min_principal_angle else 'blue', zorder=10)
                xlim_max = 4.5 if alg is eb.Algorithms.GPFA2 else 6.5
                plt.xlim(.5, xlim_max)
                #plt.ylim(-.2, np.pi/2+.2)
                plt.ylim(-5, 99)
                if idx % 4 == 0:
                    plt.ylabel('angle [deg]')
                    #labels = [item.get_text() for item in plt.gca().get_xticklabels()]
                    #labels[-1] = "$\pi/2$"
                    #plt.gca().set_yticklabels(labels)
                else:
                    plt.gca().set_yticklabels([])
                if (alg is eb.Algorithms.ForeCA and idx >= 4) or idx >= 12:
                    plt.xlabel('p')
                else:
                    plt.gca().set_xticklabels([])

            # title
            plt.title(eb.get_dataset_name(env=env, ds=dataset, latex=False), fontsize=12)

            idx += 1

        if alg is eb.Algorithms.ForeCA:
            plt.subplots_adjust(hspace=.4, wspace=.15, left=0.07, right=.96, bottom=.14, top=.84)
        else:
            plt.subplots_adjust(hspace=.4, wspace=.15, left=0.07, right=.96, bottom=.08, top=.92)
            
        plt.savefig('fig_angles_to_p1_%s.eps' % plot_alg_names[alg].lower())
    #plt.show()



if __name__ == '__main__':
    main()
    
