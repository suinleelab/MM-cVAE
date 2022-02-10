import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from glob import glob
import helper


def filter_images_by_attribute(data_dir, attr1=None, attr2=None, present1=True, present2=True):
    if attr1 is None and attr2 is None:
        return glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))
    df = pd.read_csv(os.path.join(data_dir, 'list_attr_celeba.csv'))
    assert attr1 in df.columns
    assert attr2 in df.columns
    val1 = 1 if present1 else -1
    val2 = 1 if present2 else -1
    df = df.loc[(df[attr1] == val1) & (df[attr2] == val2)]
    image_ids = df['image_id'].values
    image_ids = ['celeba_data/img_align_celeba/' + i for i in image_ids]
    return image_ids


def show_images(ids):
    show_n_images = 16
    celeb_images = helper.get_batch(ids[:show_n_images], 28, 28, 'RGB')
    plt.imshow(helper.images_square_grid(celeb_images, 'RGB'))


def plot_sweeps_celeb(decoder, latent_dim=2):
    n = 10
    digit_size = 28
    n_channels = 3
    figure = np.zeros((digit_size * n, digit_size * n, n_channels))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros((1, latent_dim))
            z_sample[0][0] = xi
            z_sample[0][1] = yi

            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, n_channels)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size, :] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)


def plot_contrastive_sweeps_celeb(decoder, latent_dim=4):
    n = 10
    digit_size = 28
    n_channels = 3
    figure = np.zeros((digit_size * n, digit_size * n, n_channels))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros((1, latent_dim))
            z_sample[0][0] = xi
            z_sample[0][1] = yi

            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, n_channels)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size, :] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros((1, latent_dim))
            z_sample[0][-1] = xi
            z_sample[0][-2] = yi

            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, n_channels)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size, :] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("s[0]")
    plt.ylabel("s[1]")
    plt.imshow(figure)


def get_synthetic_images(decoder, latent_dim=2, n=16):
    h = np.random.normal(0, 1, (n, latent_dim))
    return decoder.predict(h)
