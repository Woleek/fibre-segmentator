import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import nibabel as nib
import monai
import monai.networks.nets as mn
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.data import ImageDataset, CacheDataset, GridPatchDataset, list_data_collate,DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ResizeWithPadOrCropd,
    Spacingd,
    GridPatchd,
    Resized,
    ScaleIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ToTensord,
)
from torchmetrics.functional import accuracy
from monai.losses import DiceLoss 

class FibreSegmentator(pl.LightningModule):
    
    # computations
    def __init__(self):
        super().__init__()
        self.segmentator = mn.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64),
            strides=(2, 2),
            # num_res_units=2,
        )
        self.loss = DiceLoss(
            batch=True,
        )
        
    # forward pass
    def forward(self, x):
        out = self.segmentator(x)
        return out
    
    # data preparation
    def prepare_data(self):
        import os
        import glob
        data_path = 'data\\prep_data'
        x_list = glob.glob(os.path.join(data_path, "**\\x_prep.nii.gz"), recursive=True)
        y_list = glob.glob(os.path.join(data_path, "**\\y_prep.nii.gz"), recursive=True)

        data_dicts = [{'image':image_name, 'label':label_name} for image_name, label_name in zip(x_list, y_list)]
        
        train_dicts = data_dicts[:-1]
        val_dicts = [data_dicts[-1]]
        # test_dicts = [data_dicts[-1]]
        
        train_transforms = Compose([
            LoadImaged(
                keys=["image", "label"],
                as_closest_canonical=True,
                image_only=True,
            ),
            EnsureChannelFirstd(
                keys=["image", "label"]
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64,64,64),
                num_samples=10,
            ),
            ToTensord(
                keys=["image", "label"],
            ),
        ])
        
        val_transforms = Compose([
            LoadImaged(
                keys=["image", "label"],
            ),
            EnsureChannelFirstd(
                keys=["image", "label"]
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64,64,64),
                num_samples=10,
            ),
            ToTensord(
                keys=["image", "label"],
            ),
        ])
        
        # test_transforms = Compose([
        #     LoadImaged(
        #         keys=["image", "label"],
        #     ),
        #     EnsureChannelFirstd(
        #         keys=["image", "label"]
        #     ),
        #     ToTensord(
        #         keys=["image", "label"],
        #     ),
        # ])
        
        self.train_ds = CacheDataset(
            data=train_dicts,
            transform=train_transforms,
            cache_rate=1.0,
        )
        
        self.val_ds = CacheDataset(
            data=val_dicts,
            transform=val_transforms,
            cache_rate=1.0,
        )
        
        # self.test_ds = CacheDataset(
        #     data=test_dicts,
        #     transform=test_transforms,
        #     cache_rate=1.0,
        # )
    
    # train loader
    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=4,
        )
        return train_loader
    
    # train loop
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        
        loss = self.loss(y_hat, y)
        self.log_dict({'train_loss': loss})
        return {"loss": loss}
    
    # validation loader
    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.val_ds,
            batch_size=1,
            num_workers=4,
        )
        return val_loader
    
    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        
        loss = self.loss(y_hat, y)
        self.log_dict({'val_loss': loss}) # , on_epoch=True
        return {"loss": loss}
        
    # optimizers and lr schedulers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == '__main__':
    i = 0
    while os.path.exists(f".\\lightning_logs\\version_{i}"):
        i += 1
        
    model = FibreSegmentator()
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=10,
        devices='auto',
        fast_dev_run=False,
        logger = TensorBoardLogger(
            save_dir='.'
        ),
        callbacks=[ModelCheckpoint(
            dirpath=f".\\lightning_logs\\version_{i}",
            filename='best_checkpoint',
            save_top_k=1)],
        log_every_n_steps=1,
        auto_lr_find=True,
    )
    trainer.fit(model)
    torch.save(model, f".\\lightning_logs\\version_{i}\\model.pt")
    
    os.system('PAUSE')