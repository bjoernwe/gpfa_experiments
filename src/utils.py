import numpy as np
import sys


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


def principal_angles(A, B):
    """A and B must be column-orthogonal.
    Returns min and max principle angle.
    Golub: Matrix Computations, 1996
    [http://www.disi.unige.it/person/BassoC/teaching/python_class02.pdf]
    """
    if A.ndim == 1:
        A = np.array(A, ndmin=2).T
    if B.ndim == 1:
        B = np.array(B, ndmin=2).T
    assert A.ndim == B.ndim == 2
    A = np.linalg.qr(A)[0]
    B = np.linalg.qr(B)[0]
    _, S, _ = np.linalg.svd(np.dot(A.T, B))
    angles = np.arccos(np.clip(S, -1, 1))
    return np.min(angles), np.max(angles)


def format_arg_value(arg_val):
    """ Return a string representing a (name, value) pair.
    
    >>> format_arg_value(('x', (1, 2, 3)))
    'x=(1, 2, 3)'
    """
    arg, val = arg_val
    if isinstance(val, np.ndarray):
        return '%s=numpy.ndarray<%s>' % (arg, val.shape)
    return "%s=%r" % (arg, val)
    

def echo(fn, write=sys.stdout.write):
    """ Echo calls to a function.
    
    Returns a decorated version of the input function which "echoes" calls
    made to it by writing out the function's name and the arguments it was
    called with.
    """
    import functools
    # Unpack function's arg count, arg names, arg defaults
    code = fn.func_code
    argcount = code.co_argcount
    argnames = code.co_varnames[:argcount]
    fn_defaults = fn.func_defaults or list()
    argdefs = dict(zip(argnames[-len(fn_defaults):], fn_defaults))
    
    @functools.wraps(fn)
    def wrapped(*v, **k):
        # Collect function arguments by chaining together positional,
        # defaulted, extra positional and keyword arguments.
        positional = map(format_arg_value, zip(argnames, v))
        defaulted = [format_arg_value((a, argdefs[a]))
                     for a in argnames[len(v):] if a not in k]
        nameless = map(repr, v[argcount:])
        keyword = map(format_arg_value, k.items())
        args = positional + defaulted + nameless + keyword
        write("%s(%s)\n\n" % (fn.__name__, ", ".join(args)))
        return fn(*v, **k)
    return wrapped


def echo_on_exception(fn, write=sys.stdout.write):
    """ Echo calls to a function.
    
    Returns a decorated version of the input function which "echoes" calls
    made to it by writing out the function's name and the arguments it was
    called with.
    """
    import functools
    # Unpack function's arg count, arg names, arg defaults
    code = fn.func_code
    argcount = code.co_argcount
    argnames = code.co_varnames[:argcount]
    fn_defaults = fn.func_defaults or list()
    argdefs = dict(zip(argnames[-len(fn_defaults):], fn_defaults))
    
    @functools.wraps(fn)
    def wrapped(*v, **k):
        try:
            return fn(*v, **k)
        except Exception:
            # Collect function arguments by chaining together positional,
            # defaulted, extra positional and keyword arguments.
            positional = map(format_arg_value, zip(argnames, v))
            defaulted = [format_arg_value((a, argdefs[a]))
                         for a in argnames[len(v):] if a not in k]
            nameless = map(repr, v[argcount:])
            keyword = map(format_arg_value, k.items())
            args = positional + defaulted + nameless + keyword
            write("Threw exception: %s(%s)\n\n" % (fn.__name__, ", ".join(args)))
            raise
    return wrapped


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


@echo_on_exception
def test(a, b = 4, c = 'blah-blah', *args, **kwargs):
    assert c



def f_identity(x):
    return x



def f_exp08(x):
    return np.abs(x)**.8



def test_principle_angles():

    A = np.array([[1,0], 
                  [0,1], 
                  [0,0], 
                  [0,0]]) 
    B = np.array([[0,0], 
                  [0,1], 
                  [1,0], 
                  [0,0]])
    C = np.array([[0,0], 
                  [0,0], 
                  [1,0], 
                  [0,1]])
    
    assert principal_angles(A, A) == (0, 0)
    assert principal_angles(A, B) == (0, np.pi/2) 
    assert principal_angles(A, C) == (np.pi/2, np.pi/2)
    
    print 'okay' 



def main():
    test(1, c=False)
    #test_principle_angles()



if __name__ == '__main__':
    main()
    