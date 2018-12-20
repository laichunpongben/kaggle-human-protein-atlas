import os
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import *
from fastai.callbacks.tracker import SaveModelCallback
from config import MODEL_PATH

@dataclass
class SaveModelCustomPathCallback(SaveModelCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    every:str='improvement'
    name:str='bestmodel'
    device:str='cpu'
    def __post_init__(self):
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'
        super().__post_init__()

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        if self.every=="epoch":
            model_path = os.path.join(MODEL_PATH, "{}-epoch{}.pth".format(self.name, epoch))
            torch.save(self.learn.model.state_dict(), model_path)
        else: #every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                self.best = current
                model_path = os.path.join(MODEL_PATH, "{}.pth".format(self.name))
                torch.save(self.learn.model.state_dict(), model_path)

    def on_train_end(self, **kwargs):
        if self.every=="improvement":
            self.learn.model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "{}.pth".format(self.name)),
                                             map_location=self.device),
                                             strict=False)
