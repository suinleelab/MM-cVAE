import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from itertools import chain

class cVAE(pl.LightningModule):
    def __init__(
            self,
            input_dim,
            background_latent_size: int,
            salient_latent_size: int,
            output_activation=None
    ):
        super(cVAE, self).__init__()

        self.salient_latent_size = salient_latent_size
        self.background_latent_size = background_latent_size

        bias = True
        self.z_h = nn.Linear(input_dim, 400, bias=bias)
        self.z_mu = nn.Linear(400, background_latent_size, bias=bias)
        self.z_var = nn.Linear(400, background_latent_size, bias=bias)

        self.s_h = nn.Linear(input_dim, 400, bias=bias)
        self.s_mu = nn.Linear(400, salient_latent_size, bias=bias)
        self.s_var = nn.Linear(400, salient_latent_size, bias=bias)

        total_latent_size = background_latent_size + salient_latent_size
        self.total_latent_size = total_latent_size

        self.fc3 = nn.Linear(self.total_latent_size, 400, bias=bias)
        self.fc4 = nn.Linear(400, input_dim, bias=bias)

        self.discriminator = nn.Linear(self.total_latent_size, 1)

        self.output_activation = output_activation

    def encode(self, x):
        hz = F.relu(self.z_h(x))
        hs = F.relu(self.s_h(x))

        return self.z_mu(hz), self.z_var(hz), self.s_mu(hs), self.s_var(hs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))

        if self.output_activation == 'sigmoid':
            return torch.sigmoid(self.fc4(h3))
        else:
            return self.fc4(h3)

    def forward(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        return self.decode(torch.cat([z, s], dim=1)), mu_z, logvar_z, mu_s, logvar_s

    def forward_target(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        return self.decode(torch.cat([z, s], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s

    def forward_background(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)

        salient_var_vector = torch.zeros(x.shape[0], self.salient_latent_size).to(self.device)

        return self.decode(torch.cat([z, salient_var_vector], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s

    def embed_shared(self, x):
        mu_z, _, _, _ = self.encode(x)
        return mu_z

    def embed_salient(self, x):
        _, _, mu_s, _ = self.encode(x)
        return mu_s

    def training_step(self, batch, batch_idx):
        x, labels = batch
        background = x[labels == 0]
        targets = x[labels != 0]

        recon_batch_bg, mu_z_bg, logvar_z_bg, mu_s_bg, logvar_s_bg, z_bg, s_bg = self.forward_background(background)
        recon_batch_tar, mu_z_tar, logvar_z_tar, mu_s_tar, logvar_s_tar, z_tar, s_tar = self.forward_target(targets)

        MSE_bg = F.mse_loss(recon_batch_bg, background, reduction='sum')
        MSE_tar = F.mse_loss(recon_batch_tar, targets, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_z_bg = -0.5 * torch.sum(1 + logvar_z_bg - mu_z_bg.pow(2) - logvar_z_bg.exp())
        KLD_z_tar = -0.5 * torch.sum(1 + logvar_z_tar - mu_z_tar.pow(2) - logvar_z_tar.exp())
        KLD_s_tar = -0.5 * torch.sum(1 + logvar_s_tar - mu_s_tar.pow(2) - logvar_s_tar.exp())

        loss = (MSE_bg + KLD_z_bg) + (MSE_tar + KLD_z_tar + KLD_s_tar)

        # Corresponds to cVAE
        z1, z2 = z_tar[:int(len(z_tar) / 2)], z_tar[int(len(z_tar) / 2):]
        s1, s2 = s_tar[:int(len(s_tar) / 2)], s_tar[int(len(s_tar) // 2):]

        # In case we have an odd number of target samples
        size = min(len(z1), len(z2))
        z1, z2, s1, s2 = z1[:size], z2[:size], s1[:size], s2[:size]

        q = torch.cat([
            torch.cat([z1, s1], dim=1),
            torch.cat([z2, s2], dim=1)
        ])

        q_bar = torch.cat([
            torch.cat([z1, s2], dim=1),
            torch.cat([z2, s1], dim=1)
        ])

        q_bar_score = F.sigmoid(self.discriminator(q_bar))
        q_score = F.sigmoid(self.discriminator(q))

        tc_loss = torch.log(q_score / (1 - q_score)).sum()
        discriminator_loss = (- torch.log(q_score) - torch.log(1 - q_bar_score)).sum()

        loss += tc_loss + discriminator_loss

        self.log('tc_loss', tc_loss, prog_bar=True)
        self.log('discriminator_loss', discriminator_loss, prog_bar=True)

        self.log('MSE_bg', MSE_bg, prog_bar=True)
        self.log('MSE_tar', MSE_tar, prog_bar=True)
        self.log('KLD_z_bg', KLD_z_bg, prog_bar=True)
        self.log('KLD_z_tar', KLD_z_tar, prog_bar=True)
        self.log('KLD_s_tar', KLD_s_tar, prog_bar=True)

        return loss

    def configure_optimizers(self):

        params = chain(
            self.z_h.parameters(),
            self.z_mu.parameters(),
            self.z_var.parameters(),
            self.s_h.parameters(),
            self.s_mu.parameters(),
            self.s_var.parameters(),
            self.fc3.parameters(),
            self.fc4.parameters(),
            self.discriminator.parameters()
        )

        opt = torch.optim.Adam(params)
        return opt


class Conv_cVAE(pl.LightningModule):
    def __init__(self):
        super(Conv_cVAE, self).__init__()
        in_channels = 3
        dec_channels = 32
        salient_latent_size = 6
        background_latent_size = 16
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.salient_latent_size = salient_latent_size
        self.background_latent_size = background_latent_size
        bias = False

        self.z_convs = nn.Sequential(
            nn.Conv2d(in_channels, dec_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),


            nn.Conv2d(dec_channels, dec_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels*2),


            nn.Conv2d(dec_channels * 2, dec_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),

            nn.Conv2d(dec_channels * 4, dec_channels * 8, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 8)
        )

        self.s_convs = nn.Sequential(
            nn.Conv2d(in_channels, dec_channels, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),

            nn.Conv2d(dec_channels, dec_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 2),

            nn.Conv2d(dec_channels * 2, dec_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),

            nn.Conv2d(dec_channels * 4, dec_channels * 8, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 8)
        )

        self.z_mu = nn.Linear(dec_channels * 8 * 4 * 4, background_latent_size, bias=bias)
        self.z_var = nn.Linear(dec_channels * 8 * 4 * 4, background_latent_size, bias=bias)

        self.s_mu = nn.Linear(dec_channels * 8 * 4 * 4, salient_latent_size, bias=bias)
        self.s_var = nn.Linear(dec_channels * 8 * 4 * 4, salient_latent_size, bias=bias)

        self.decode_convs = nn.Sequential(
            nn.ConvTranspose2d(dec_channels * 8, dec_channels * 4,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),

            nn.ConvTranspose2d(dec_channels * 4, dec_channels * 2,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 2),

            nn.ConvTranspose2d(dec_channels * 2, dec_channels,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),

            nn.ConvTranspose2d(dec_channels, in_channels,
                               kernel_size=(4, 4), stride=(2, 2), padding=1, bias=bias),
            nn.Sigmoid()
        )

        total_latent_size = salient_latent_size + background_latent_size

        self.discriminator = nn.Linear(total_latent_size, 1)
        self.d_fc_1 = nn.Linear(total_latent_size, dec_channels * 8 * 4 * 4)

    def reparameterize(self, mu, log_var):
        #:param mu: mean from the encoder's latent space
        #:param log_var: log variance from the encoder's latent space
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def encode(self, x):
        hz = self.z_convs(x)
        hs = self.s_convs(x)

        hz = hz.view(-1, self.dec_channels * 8 * 4 * 4)
        hs = hs.view(-1, self.dec_channels * 8 * 4 * 4)

        return self.z_mu(hz), self.z_var(hz), self.s_mu(hs), self.s_var(hs)

    def decode(self, z):
        z = F.leaky_relu(self.d_fc_1(z), negative_slope=0.2)
        z = z.view(-1, self.dec_channels * 8, 4, 4)

        return self.decode_convs(z)

    def forward_target(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        return self.decode(torch.cat([z, s], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s

    def forward_background(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        salient_var_vector = torch.zeros(x.shape[0], self.salient_latent_size).to(self.device)
        return self.decode(torch.cat([z, salient_var_vector], dim=1)), mu_z, logvar_z, mu_s, logvar_s, z, s

    def embed_shared(self, x):
        mu_z, _, _, _ = self.encode(x)
        return mu_z

    def embed_salient(self, x):
        _, _, mu_s, _ = self.encode(x)
        return mu_s

    def forward(self, x):
        mu_z, logvar_z, mu_s, logvar_s = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        s = self.reparameterize(mu_s, logvar_s)
        return self.decode(torch.cat([z, s], dim=1)), mu_z, logvar_z, mu_s, logvar_s

    def training_step(self, batch, batch_idx):
        x, labels = batch
        background = x[labels == 0]
        targets = x[labels != 0]

        recon_batch_bg, mu_z_bg, logvar_z_bg, mu_s_bg, logvar_s_bg, z_bg, s_bg = self.forward_background(background)
        recon_batch_tar, mu_z_tar, logvar_z_tar, mu_s_tar, logvar_s_tar, z_tar, s_tar = self.forward_target(targets)

        MSE_bg = F.mse_loss(recon_batch_bg, background, reduction='sum')
        MSE_tar = F.mse_loss(recon_batch_tar, targets, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_z_bg = -0.5 * torch.sum(1 + logvar_z_bg - mu_z_bg.pow(2) - logvar_z_bg.exp())
        KLD_z_tar = -0.5 * torch.sum(1 + logvar_z_tar - mu_z_tar.pow(2) - logvar_z_tar.exp())
        KLD_s_tar = -0.5 * torch.sum(1 + logvar_s_tar - mu_s_tar.pow(2) - logvar_s_tar.exp())

        loss = (MSE_bg + KLD_z_bg) + (MSE_tar + KLD_z_tar + KLD_s_tar)

        z1, z2 = z_tar[:int(len(z_tar) / 2)], z_tar[int(len(z_tar) / 2):]
        s1, s2 = s_tar[:int(len(s_tar) / 2)], s_tar[int(len(s_tar) // 2):]

        # In case we have an odd number of target samples
        size = min(len(z1), len(z2))
        z1, z2, s1, s2 = z1[:size], z2[:size], s1[:size], s2[:size]

        q = torch.cat([
            torch.cat([z1, s1], dim=1),
            torch.cat([z2, s2], dim=1)
        ])

        q_bar = torch.cat([
            torch.cat([z1, s2], dim=1),
            torch.cat([z2, s1], dim=1)
        ])

        q_bar_score = F.sigmoid(self.discriminator(q_bar))
        q_score = F.sigmoid(self.discriminator(q))

        eps = 1e-6
        q_score = q_score.clone().where(q_score == 0, torch.tensor(eps).to(self.device))
        q_score = q_score.clone().where(q_score == 1, torch.tensor(1 - eps).to(self.device))

        q_bar_score = q_bar_score.clone().where(q_bar_score == 0, torch.tensor(eps).to(self.device))
        q_bar_score = q_bar_score.clone().where(q_bar_score == 1, torch.tensor(1 - eps).to(self.device))

        tc_loss = torch.log(q_score / (1 - q_score)).sum()
        discriminator_loss = (- torch.log(q_score) - torch.log(1 - q_bar_score)).sum()

        loss += tc_loss + discriminator_loss

        self.log('tc_loss', tc_loss, prog_bar=True)
        self.log('discriminator_loss', discriminator_loss, prog_bar=True)

        self.log('MSE_bg', MSE_bg, prog_bar=True)
        self.log('MSE_tar', MSE_tar, prog_bar=True)
        self.log('KLD_z_bg', KLD_z_bg, prog_bar=True)
        self.log('KLD_z_tar', KLD_z_tar, prog_bar=True)
        self.log('KLD_s_tar', KLD_s_tar, prog_bar=True)

        return loss

    def configure_optimizers(self):

        params = chain(
            self.z_convs.parameters(),
            self.s_convs.parameters(),
            self.z_mu.parameters(),
            self.z_var.parameters(),
            self.s_mu.parameters(),
            self.s_var.parameters(),
            self.d_fc_1.parameters(),
            self.decode_convs.parameters(),
            self.discriminator.parameters()
        )

        opt = torch.optim.Adam(params)
        return opt
