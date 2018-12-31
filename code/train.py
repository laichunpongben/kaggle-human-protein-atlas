from fastai.torch_core import *
from fastai.callbacks import *
from fastai.basic_data import *
from fastai.basic_train import *

def my_fit_one_cycle(learn:Learner, cyc_len:int, max_lr:Union[Floats,slice]=defaults.lr,
                  moms:Tuple[float,float]=(0.95,0.85), div_factor:float=25., pct_start:float=0.3,
                  wd:float=None, callbacks:Optional[CallbackList]=None, **kwargs)->None:
    "Fit a model following the 1cycle policy."
    max_lr = learn.lr_range(max_lr)
    print(max_lr)
    callbacks = ifnone(callbacks, [])
    callbacks.append(OneCycleScheduler(learn, max_lr, moms=moms, div_factor=div_factor,
                                        pct_start=pct_start, **kwargs))
    learn.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks)
