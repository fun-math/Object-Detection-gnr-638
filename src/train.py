from pl_train import YOLOv4PL


import pytorch_lightning as pl
from argparse import Namespace
from pytorch_lightning.callbacks import LearningRateLogger


hparams = {
    "pretrained" : False,
    "train_ds" : "../dataset/train.txt",
    "valid_ds" : "../dataset/valid.txt",
    "bs" : 1,
    "momentum": 0.9,
    "wd": 0.001,
    "lr": 1e-8,
    "epochs" : 100,
    "pct_start" : 10/100,
    "optimizer" : "SGD",
    "SAT" : False,
    "epsilon" : 0.1,
    "Dropblock" : False,
    "scheduler" : "Cosine Warm-up"
}

hparams = Namespace(**hparams)
m = YOLOv4PL(hparams)

tb_logger = pl.loggers.TensorBoardLogger('logs/', name = "yolov4")

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath='model_checkpoints/yolov4{epoch:02d}',
    verbose=True,
    monitor="training_loss_epoch",
    mode='min',
)

t = pl.Trainer(logger = tb_logger,
           checkpoint_callback=checkpoint_callback,
           gpus=1,
           precision=32,
           benchmark=True,
           callbacks=[LearningRateLogger()],
           min_epochs=100,


#            resume_from_checkpoint="model_checkpoints/yolov4epoch=82.ckpt",
        #    auto_lr_find=True,
          #  auto_scale_batch_size='binsearch',
        #    fast_dev_run=True
          )



r = t.lr_find(m, min_lr=1e-10, max_lr=1e-3, early_stop_threshold=None)
r.plot()

t.fit(m)
