from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import mkl
import numpy as np
import scipy.stats

#import experiments_proxy.experiment_base_proxy as eb
import experiment_base as eb
import parameters



def main():
    
    mkl.set_num_threads(1)
    
    plot_alg_names = {eb.Algorithms.Random: 'random',
                      eb.Algorithms.SFA:    'SFA',
                      eb.Algorithms.SFFA:   "SFA'",
                      eb.Algorithms.ForeCA: 'ForeCA',
                      eb.Algorithms.PFA:    'PFA',
                      eb.Algorithms.GPFA2:  'GPFA',
                      }
    
    algs = [eb.Algorithms.SFA,
            #eb.Algorithms.ForeCA,
            eb.Algorithms.PFA,
            #eb.Algorithms.GPFA2
            ]

    results = {}
    results_pca = {}
    for alg in algs:
        print(alg)
        #only_low_dimensional = alg is eb.Algorithms.ForeCA
        results[alg] = parameters.get_results(alg)#, overide_args={'use_test_set': False})
        results_pca[alg] = parameters.get_results(alg, overide_args={'algorithm': eb.Algorithms.PCA}) #, 'use_test_set': False})

    for alg in algs:
        
        colors = iter(matplotlib.cm.get_cmap('pink')(np.linspace(0, 1, int(1.25*len(parameters.dataset_args)))))
        markers = iter(['*', 'o', '^', 'v', '<', '>', 'd', 'D', 's'] * 2)
        
        plt.figure(figsize=(10,6))

        for dataset_args in parameters.dataset_args:
            
            env = dataset_args['env']
            dataset = dataset_args['dataset']
            if not dataset in results[alg]:
                continue
            result = results[alg][dataset].values
            result_pca = results_pca[alg][dataset].values
            
            if True:
                # average over first dim (output_dim)
                result = np.mean(result, axis=0, keepdims=True) 
                result_pca = np.mean(result_pca, axis=0, keepdims=True)
            
            # point cloud
            color = next(colors)
            marker = next(markers)
            #for i in range(result.shape[0]):
            #    plt.scatter(result[i], result_sfa[i], c=color, marker=marker, label=None, s=80, alpha=.2, linewidths=0, zorder=1)
    
            # plot
            mu = np.mean(result, axis=-1) # last axis = repetitions
            values0 = (result.T - mu).T
            values0_dummy = np.array(values0, copy=True)
            values0_dummy[values0 < 0] = np.NaN
            errors_pos = np.sqrt(np.nanmean(values0_dummy**2, axis=-1))
            values0_dummy = np.array(values0, copy=True)
            values0_dummy[values0 > 0] = np.NaN
            errors_neg = np.sqrt(np.nanmean(values0_dummy**2, axis=-1))
    
            mu_pca = np.mean(result_pca, axis=-1) # 1st axis = output_dim, last axis = repetitions
            values0_pca = (result_pca.T - mu_pca).T
            values0_pca_dummy = np.array(values0_pca, copy=True)
            values0_pca_dummy[values0_pca < 0] = np.NaN
            errors_pca_pos = np.sqrt(np.nanmean(values0_pca_dummy**2, axis=-1))
            values0_sfa_dummy = np.array(values0_pca, copy=True)
            values0_sfa_dummy[values0_pca > 0] = np.NaN
            errors_pca_neg = np.sqrt(np.nanmean(values0_pca_dummy**2, axis=-1))
    
            label = '%s' % eb.get_dataset_name(env=env, ds=dataset, latex=False) #%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            xerr = np.vstack([errors_neg, errors_pos])
            yerr = np.vstack([errors_pca_neg, errors_pca_pos])
            plt.errorbar(mu, mu_pca, xerr=xerr, yerr=yerr, c=color, marker=marker, markersize=9, label=label, zorder=2)
            
        # 
        measure_label = 'prediction errors on'
        if alg is eb.Algorithms.ForeCA:
            measure_label = 'forcastability of'
        elif alg is eb.Algorithms.Random or alg is eb.Algorithms.SFA:
            measure_label = 'delta values'
        measure_limits = [1e0, 1e2] if alg is eb.Algorithms.ForeCA else [1e-4, 1e2]
        plt.plot(measure_limits, measure_limits, '-', c='gray', zorder=3)
        plt.suptitle(plot_alg_names[alg])
        plt.xlabel('%s %s features' % (measure_label, plot_alg_names[alg]))
        plt.ylabel('%s PCA features' % (measure_label))
        plt.xscale('log')
        plt.yscale('log')
        handles, labels = plt.gca().get_legend_handles_labels()
        handles = [h[0] for h in handles]
        plt.legend(handles, labels, loc='best', prop={'size': 9}, numpoints=1, borderpad=1, handlelength=0)
        plt.tight_layout()
        plt.savefig('fig_results_random_%s.eps' % plot_alg_names[alg].lower())

    plt.show()



if __name__ == '__main__':
    main()
    
