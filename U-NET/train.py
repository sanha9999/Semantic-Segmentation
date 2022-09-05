from msilib.schema import MsiPatchHeaders
import os
from unicodedata import east_asian_width
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser
import numpy as np
import torch

from trainer import UnetModel, UnetDataModule

def define_argparser():
    p = ArgumentParser(add_help=False)

    p.add_argument('--dataset', required=False, default='C:/Users/kangsanha/Desktop/segmentation/cityscapes_data')
    p.add_argument('--log_dir', default='lightning_logs')

    parser = UnetModel.add_model_specific_args(parent_parser=p)

    params = parser.parse_args()

    return params

def main(params):
    working_dir = 'C:/Users/kangsanha/Desktop/segmentation/U-NET/'
    model = UnetModel(params=params)
    img_dir = params.dataset

    dm = UnetDataModule(img_dir=img_dir)

    #os.makedirs(working_dir + params.log_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath= working_dir + params.log_dir,
        filename='unet-{epoch}-{val_loss:.2f}'
    )

    stop_callback = EarlyStopping(
        monitor='val_loss',
    )

    trainer = Trainer(
        accelerator="auto", 
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=10,
        callbacks=[checkpoint_callback ,stop_callback],
    )
    
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    params = define_argparser()
    main(params)