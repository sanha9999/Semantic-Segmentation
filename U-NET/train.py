import wandb
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
import numpy as np
import torch
from pytorch_lightning.loggers import WandbLogger

from trainer import UnetModel, UnetDataModule

def define_argparser():
    p = ArgumentParser(add_help=False)

    p.add_argument('--dataset', required=False, default='C:/Users/kangsanha/Desktop/segmentation/cityscapes_data')
    p.add_argument('--log_dir', default='lightning_logs')

    parser = UnetModel.add_model_specific_args(parent_parser=p)

    params = parser.parse_args()

    return params

def main(params):
    wandb.login()

    working_dir = 'C:/Users/kangsanha/Desktop/segmentation/U-NET/'
    model = UnetModel(params=params)
    img_dir = params.dataset
    n_classes = params.n_classes

    dm = UnetDataModule(img_dir=img_dir, n_classes = n_classes)

    wandb_logger = WandbLogger(name='Adam-epoch50-0.005', project='pytorchlightning')

    #os.makedirs(working_dir + params.log_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath= working_dir + params.log_dir,
        filename='unet-0.005-{epoch}-{val_loss:.2f}',
        verbose=True
    )

    stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True
    )

    trainer = Trainer(
        accelerator="auto", 
        devices=1 if torch.cuda.is_available() else None,
        logger=wandb_logger,
        max_epochs=50,
        callbacks=[checkpoint_callback, stop_callback],
    )
    
    trainer.fit(model, datamodule=dm)
    #wandb_logger.watch(model)

if __name__ == '__main__':
    params = define_argparser()
    main(params)