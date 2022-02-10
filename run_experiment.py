import argparse
import os

import numpy as np
import pytorch_lightning as pl
import torch
from data import SimpleDataset
from models.sRB_VAE import sRB_VAE, Conv_sRB_VAE
from models.cVAE import cVAE, Conv_cVAE
from models.MM_cVAE import MM_cVAE, Conv_MM_cVAE
from torch.utils.data import DataLoader
from data import load_aml, load_epithel
import helper
from data import CelebADataset

from utils import set_seeds

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['Epithel', 'CelebA', 'AML'])
parser.add_argument('model', type=str, choices=['mm_cvae', 'cvae', 'srb_vae'])

args = parser.parse_args()

set_seeds()

if args.dataset == 'Epithel':
    (data1, labels1), (data2, labels2), (data3, labels3) = load_epithel()

    data_total = np.concatenate([data1, data2, data3])
    labels_total = np.concatenate([labels1, labels2, labels3])

    dataset = SimpleDataset(X=data_total, y=labels_total)
    epochs = 100
    trainer = pl.Trainer(max_epochs=epochs, gpus=[0])
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=12)

    if args.model == 'srb_vae':
        model = sRB_VAE(
            input_dim=data1.shape[1],
            background_latent_size=10,
            salient_latent_size=5
        )
    elif args.model == 'cvae':
        model = cVAE(
            input_dim=data1.shape[1],
            background_latent_size=10,
            salient_latent_size=5,
        )

    elif args.model == 'mm_cvae':
        model = MM_cVAE(
            input_dim=data1.shape[1],
            background_latent_size=10,
            salient_latent_size=5,
            background_disentanglement_penalty=10e3,
            salient_disentanglement_penalty=10e2
        )

    trainer.fit(model, loader)

elif args.dataset == 'AML':
    (data1, labels1), (data2, labels2), (data3, labels3) = load_aml()

    data_total = np.concatenate([data1, data2, data3])
    labels_total = np.concatenate([labels1, labels2, labels3])

    dataset = SimpleDataset(X=data_total, y=labels_total)
    epochs = 100
    trainer = pl.Trainer(max_epochs=epochs, gpus=[0])
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=12)

    if args.model == 'srb_vae':
        model = sRB_VAE(
            input_dim=data1.shape[1],
            background_latent_size=10,
            salient_latent_size=5
        )
    elif args.model == 'cvae':
        model = cVAE(
            input_dim=data1.shape[1],
            background_latent_size=10,
            salient_latent_size=5
        )

    elif args.model == 'mm_cvae':
        model = MM_cVAE(
            input_dim=data1.shape[1],
            background_latent_size=10,
            salient_latent_size=5,
            background_disentanglement_penalty=10e3,
            salient_disentanglement_penalty=10e2
        )


    trainer.fit(model, loader)

elif args.dataset == 'CelebA':
    data_dir = 'celeba_data/'
    helper.download_extract('celeba', data_dir)

    total_ids = np.load("celeba_ids.npy")
    total_labels = np.load("celeba_labels.npy")

    dataset = CelebADataset(
        total_ids,
        labels=total_labels
    )

    epochs = 100
    trainer = pl.Trainer(max_epochs=epochs, gpus=1)

    loader = DataLoader(dataset, batch_size=128, num_workers=12, shuffle=True)

    if args.model == 'srb_vae':
        model = Conv_sRB_VAE()
    elif args.model == 'cvae':
        model = Conv_cVAE()
    elif args.model == 'mm_cvae':
        model = Conv_MM_cVAE(
            background_disentanglement_penalty=10e3,
            salient_disentanglement_penalty=10e2
        )

    trainer.fit(model, loader)

checkpoint_dir = "results/" + args.dataset + "/" + args.model + "/"
os.makedirs(checkpoint_dir, exist_ok=True)
torch.save(model, checkpoint_dir + "checkpoint.chkpt")
