from __future__ import print_function

import numpy as np

import envs.env_random

import experiment_base as eb
import parameters



def main():

    results = parameters.get_results(alg=eb.Algorithms.SFA, overide_args={'measure': eb.Measures.ndims})

    f = open('./table_dims.tex', 'w+')
    print("""
\\begin{center}
\\begin{tabular}{L{5cm} R{1.25cm} R{1.25cm} R{1.25cm} R{1.25cm} R{1.25cm}}
\\toprule 
Dataset & $S$ & $S_\\textrm{train}$ & $S_\\textrm{test}$ & $N$ & $N'$ \\\\
\\midrule""", file=f)
    for dataset_args in parameters.dataset_args:

        env = dataset_args['env']
        dataset = dataset_args['dataset']
        #if dataset is None:
        #    continue
        
        kwargs = dict(parameters.default_args_global)
        #kwargs['algorithm'] = alg
        #kwargs['measure'] = algorithm_measures[alg]
        kwargs.update(dataset_args)
        #kwargs.update(parameters.dataset_default_args.get(dataset, {}))
        #kwargs.update(algorithm_parameters.get(alg, {}).get(dataset, {}))
        #kwargs.update(parameters.algorithm_args.get(alg, {}))
        #kwargs.update(overide_args)

        if dataset is None:
            print('\\midrule\n', file=f)
        print('\\texttt{%s}' % eb.get_dataset_name(env=env, ds=dataset, latex=True), end='', file=f)
        print(' & ', end='', file=f)
        if env is envs.env_random.EnvRandom:
            e = env(ndim=kwargs['ndim'])
            N, M = None, e.ndim
        else:
            e = env(dataset, limit_data=parameters.default_args_global['limit_data'], seed=0)
            N, M = e.data.shape
        if dataset is None:
            print('', end='', file=f)
        else:
            print('%d' % N, end='', file=f)
        print(' & ', end='', file=f)
        print('%d' % kwargs['n_train'], end='', file=f)
        print(' & ', end='', file=f)
        print('%d' % kwargs['n_test'], end='', file=f)
        print(' & ', end='', file=f)
        print('%d' % M, end='', file=f)
        print(' & ', end='', file=f)
        dim_avg = np.mean(results[dataset].values) # axis 0 = output_dim
        dim_std = np.std(results[dataset].values) # axis 0 = output_dim
        if kwargs['pca'] < 1.0:
            #print('%0.1f $\pm$ %0.1f' % (dim_avg, dim_std), end='', file=f)
            print('$\sim$%d' % int(round(dim_avg)), end='', file=f)
        else:
            print('%d' % dim_avg, end='', file=f)
        print(' \\\\\n', file=f)
    #print('\\texttt{MISC\\_NOISE} & & 10000 & 2000 & 20 & 20 \\\\', file=f)
    print('\\bottomrule', file=f)
    print('\\end{tabular}\n\\end{center}', file=f)



if __name__ == '__main__':
    main()
    
