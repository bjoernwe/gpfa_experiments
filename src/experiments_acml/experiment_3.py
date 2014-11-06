import numpy as np

from matplotlib import pyplot

import mdp

import fpp

from envs.env_swiss_roll_3d import EnvSwissRoll3D

if __name__ == '__main__':

    # parameters
    k = 5
    N = 5000
    whitening = True
    iterations = 1
    expansion_degree = 2
    seed = None

    # data
    env = EnvSwissRoll3D()
    data, _, labels = env.do_random_steps(num_steps=N)

    # expansion
    expansion = mdp.nodes.PolynomialExpansionNode(degree=expansion_degree)
    data = expansion.execute(data)

    # whitening
    if whitening:
        whitening_node = mdp.nodes.WhiteningNode()
        whitening_node.train(data)
        data = whitening_node.execute(data)

    # algorithms
    model = fpp.FPP(output_dim=2,
                    k=k,
                    iterations=iterations,
                    iteration_dim=2)

    # train & plot

    # current algorithm
    print "%s" % (model.__class__.__name__)

    # train
    model.train(data)
    result = model.execute(data)

    # plot
    pyplot.scatter(x=result[1:,0], y=result[1:,1], c=labels, s=50, 
                   cmap=pyplot.cm.get_cmap('Blues'))
    pyplot.title(model.__class__.__name__)

    # show plot
    print 'finish'
    pyplot.show()