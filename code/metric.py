def acc(preds,targs,th=0.0):
    '''
    Prevent the following RuntimeError:
    Expected object of scalar type Long but got scalar type Float for argument #2 'other'
    '''
    preds = (preds > th).int()
    targs = targs.int()
    return (preds==targs).float().mean()
