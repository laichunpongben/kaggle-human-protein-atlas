import os
from fastai.torch_core import *
from fastai.callback import *
from fastai.basic_train import *
from fastai.callbacks.tracker import SaveModelCallback
from fastai.callbacks.csv_logger import CSVLogger

from config import MODEL_PATH, LOG_PATH

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
            model_path = Path(MODEL_PATH)/f'{self.name}-epoch{epoch}.pth'
            torch.save(self.learn.model.state_dict(), model_path)
        else: #every="improvement"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                self.best = current
                model_path = Path(MODEL_PATH)/f'{self.name}.pth'
                torch.save(self.learn.model.state_dict(), model_path)

    def on_train_end(self, **kwargs):
        if self.every=="improvement":
            model_path = Path(MODEL_PATH)/f'{self.name}-epoch{epoch}.pth'
            self.learn.model.load_state_dict(torch.load(model_path,
                                                        map_location=self.device),
                                             strict=False)

@dataclass
class CSVCustomPathLogger(CSVLogger):
    "A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`."
    filename: str = 'history'

    def __post_init__(self):
        super().__post_init__()
        self.path = Path(LOG_PATH)/f'{self.filename}.csv'

    def read_logged_file(self):
        "Read the content of saved file"
        return pd.read_csv(self.path)

    def on_train_begin(self, metrics_names: StrList, **kwargs: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('w')
        self.file.write(','.join(self.learn.recorder.names) + '\n')

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        last_metrics = ifnone(last_metrics, [])
        stats = [str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                 for name, stat in zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics)]
        str_stats = ','.join(stats)
        self.file.write(str_stats + '\n')

    def on_train_end(self, **kwargs: Any) -> None:  self.file.close()
