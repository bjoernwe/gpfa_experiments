import collections
import functools
import numpy as np
import multiprocessing


def my_func(a, b, c=False):
    N = 3000
    A = np.random.randn(N, N)
    _, _ = np.linalg.eig(A)
    print 'a:', a
    print 'b:', b
    print 'c:', c
    print '\n'


"""
A simple wrapper for function f that allows having a specific argument
('arg_name') as the first argument.
"""
def f_wrapper(arg, arg_name, f, **kwargs):
    kwargs[arg_name] = arg
    print kwargs
    return f(**kwargs)


"""
Plots the real-valued function f using its given arguments. One of the argument
is expected to be an iterable, which is used for the x-axis.
"""
def plot(f, **kwargs):
    iterable_arguments = [k for (k, v) in kwargs.items() 
                          if isinstance(v, collections.Iterable)]
    if len(iterable_arguments) == 0:
        print 'Warning: No iterable argument found for plotting.'
        return
    elif len(iterable_arguments) >= 2:
        print 'Warning: More than one iterable argument found for plotting.'
        return
    else:
        arg_name = iterable_arguments[0]
        arg = kwargs.pop(arg_name)
        f_partial = functools.partial(f_wrapper, arg_name=arg_name, f=f,
                                      **kwargs)
        pool = multiprocessing.Pool()
        result = pool.map(f_partial, arg)
        pool.close()
        pool.join()
    return result


def main():
    plot(my_func, a=-1, b=[1, 2, 3, 4, 5], c=True)


if __name__ == '__main__':
    main()
