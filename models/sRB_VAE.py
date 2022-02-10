import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import chain

class sRB_VAE(pl.LightningModule):
    def __init__(
            self,
            input_dim,
            background_latent_size: int,
            salient_latent_size: int,
            output_activation=None
    ):
        super(sRB_VAE, self).__init__()

        self.salient_latent_size = salient_latent_size
        self.background_latent_size = background_latent_size
        total_latent_size = background_latent_size + salient_latent_size
        self.total_latent_size = total_latent_size

        self.z_h = nn.Linear(input_dim, 400)
        self.z_mu = nn.Linear(400, total_latent_size)
        self.z_var = nn.Linear(400, total_latent_size)


        self.fc3 = nn.Linear(self.total_latent_size, 400)
        self.fc4 = nn.Linear(400, input_dim)

        self.discriminator_reference = nn.Linear(input_dim + background_latent_size, 1)
        self.discriminator_target = nn.Linear(input_dim + total_latent_size, 1)
        self.reference_vector = torch.rand(salient_latent_size, requires_grad=True)

        self.output_activation = output_activation

    def encode(self, x):
        hz = F.relu(self.z_h(x))
        all_mus = self.z_mu(hz)
        all_vars = self.z_var(hz)

        z_mu, s_mu = all_mus[:, :self.background_latent_size], all_mus[:, self.background_latent_size:]
        z_var, s_var = all_vars[:, :self.background_latent_size], all_vars[:, self.background_latent_size:]

        return z_mu, z_var, s_mu, s_var

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

        salient_var_vector = self.reference_vector.repeat(z.shape[0]).reshape(z.shape[0], -1).to(self.device)

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

        # Corresponds to sRB-VAE

        # Loss term "a"
        # Want to discriminate between "real" target data points and sampled ones
        inputs_real = torch.cat([targets, z_tar, s_tar], dim=1)

        eps = torch.randn((targets.shape[0], self.total_latent_size)).to(self.device)
        inputs_fake = torch.cat([self.decode(eps), eps], dim=1)

        total_inputs = torch.cat([inputs_real, inputs_fake], dim=0)
        total_labels = torch.cat(
            [torch.zeros(inputs_real.shape[0]),
             torch.ones(inputs_fake.shape[0])], dim=0).to(self.device).reshape(-1, 1)
        discriminator_outputs = F.sigmoid(self.discriminator_target(total_inputs))
        loss_a = F.binary_cross_entropy(input=discriminator_outputs, target=total_labels, reduction='sum')

        # Loss term "b"
        # Want to discriminate between "real" background data points and sampled ones
        inputs_real = torch.cat([background, z_bg], dim=1)
        eps = torch.randn((background.shape[0], self.background_latent_size)).to(self.device)
        inputs_fake = torch.cat([
            self.decode(
                torch.cat([eps, self.reference_vector.repeat(eps.shape[0]).reshape(eps.shape[0], -1).to(self.device)], dim=1)),
            eps], dim=1)

        total_inputs = torch.cat([inputs_real, inputs_fake], dim=0)
        total_labels = torch.cat(
            [torch.zeros(inputs_real.shape[0]),
             torch.ones(inputs_fake.shape[0])], dim=0).to(self.device).reshape(-1, 1)
        discriminator_outputs = F.sigmoid(self.discriminator_reference(total_inputs))

        loss_b = F.binary_cross_entropy(input=discriminator_outputs, target=total_labels, reduction='sum')

        # Loss term "c"
        # Reconstruction error for target samples

        loss_c = MSE_tar

        # Loss term "d"
        # Reconstruction error for background samples

        loss_d = MSE_bg

        # Loss term "e"
        # Latent variable reconstruction error for target samples
        eps = torch.randn((targets.shape[0], self.total_latent_size)).to(self.device)
        inputs_fake = self.decode(eps)
        _, _, _, _, _, z_tar, s_tar = self.forward_target(inputs_fake)
        eps_fake = torch.cat([z_tar, s_tar], dim=1)
        loss_e = F.mse_loss(eps, eps_fake, reduction='sum')

        # Loss term "f"
        # Latent variable reconstruction error for background samples
        eps = torch.randn((targets.shape[0], self.background_latent_size)).to(self.device)
        inputs_fake = self.decode(torch.cat([eps, self.reference_vector.repeat(eps.shape[0]).reshape(eps.shape[0], -1).to(self.device)], dim=1))
        _, _, _, _, _, z_bg, s_bg = self.forward_background(inputs_fake)
        eps_fake = z_bg
        loss_f = F.mse_loss(eps, eps_fake, reduction='sum')

        # Total loss
        loss = loss_a + loss_b + loss_c + loss_d + loss_e + loss_f

        self.log('loss_a', loss_a, prog_bar=True)
        self.log('loss_b', loss_b, prog_bar=True)
        self.log('loss_c', loss_c, prog_bar=True)
        self.log('loss_d', loss_d, prog_bar=True)
        self.log('loss_e', loss_e, prog_bar=True)
        self.log('loss_f', loss_f, prog_bar=True)

        return loss

    def configure_optimizers(self):

        params = chain(
            self.z_h.parameters(),
            self.z_mu.parameters(),
            self.z_var.parameters(),
            self.fc3.parameters(),
            self.fc4.parameters(),
            self.discriminator_reference.parameters(),
            self.discriminator_target.parameters(),
            [self.reference_vector]
        )

        opt = torch.optim.Adam(params)
        return opt

class Conv_sRB_VAE(pl.LightningModule):
    def __init__(self):
        super(Conv_sRB_VAE, self).__init__()
        in_channels = 3
        dec_channels = 32
        salient_latent_size = 6
        background_latent_size = 16
        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.salient_latent_size = salient_latent_size
        self.background_latent_size = background_latent_size
        total_latent_size = salient_latent_size + background_latent_size

        self.z_convs = nn.Sequential(
            nn.Conv2d(in_channels, dec_channels,
                      kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),


            nn.Conv2d(dec_channels, dec_channels * 2,
                      kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels*2),


            nn.Conv2d(dec_channels * 2, dec_channels * 4,
                      kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),

            nn.Conv2d(dec_channels * 4, dec_channels * 8,
                                  kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 8)
        )

        self.z_mu = nn.Linear(dec_channels * 8 * 4 * 4, total_latent_size)
        self.z_var = nn.Linear(dec_channels * 8 * 4 * 4, total_latent_size)

        self.decode_convs = nn.Sequential(
            nn.ConvTranspose2d(dec_channels * 8, dec_channels * 4,
                               kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),

            nn.ConvTranspose2d(dec_channels * 4, dec_channels * 2,
                               kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 2),

            nn.ConvTranspose2d(dec_channels * 2, dec_channels,
                               kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),

            nn.ConvTranspose2d(dec_channels, in_channels,
                               kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.Sigmoid()
        )

        self.discriminator_reference_convs = nn.Sequential(
            nn.Conv2d(in_channels, dec_channels,
                      kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),

            nn.Conv2d(dec_channels, dec_channels * 2,
                      kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 2),

            nn.Conv2d(dec_channels * 2, dec_channels * 4,
                      kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),
            nn.Flatten()
        )

        self.discriminator_target_convs = nn.Sequential(
            nn.Conv2d(in_channels, dec_channels,
                      kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels),

            nn.Conv2d(dec_channels, dec_channels * 2,
                      kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 2),

            nn.Conv2d(dec_channels * 2, dec_channels * 4,
                      kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(dec_channels * 4),
            nn.Flatten()
        )

        # Hardcoded here based on convolutional architecture
        self.discriminator_target = nn.Linear(8192 + total_latent_size, 1)
        self.discriminator_reference = nn.Linear(8192 + background_latent_size, 1)
        self.reference_vector = torch.rand(salient_latent_size, requires_grad=True)
        self.total_latent_size = self.background_latent_size + self.salient_latent_size

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
        hz = hz.view(-1, self.dec_channels * 8 * 4 * 4)

        all_mus = self.z_mu(hz)
        all_vars = self.z_var(hz)

        z_mu, s_mu = all_mus[:, :self.background_latent_size], all_mus[:, self.background_latent_size:]
        z_var, s_var = all_vars[:, :self.background_latent_size], all_vars[:, self.background_latent_size:]

        return z_mu, z_var, s_mu, s_var

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
        salient_var_vector = self.reference_vector.repeat(z.shape[0]).reshape(z.shape[0], -1).to(self.device)

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

        # Corresponds to sRB-VAE

        # Loss term "a"
        # Want to discriminate between "real" target data points and sampled ones
        inputs_real_convs = self.discriminator_target_convs(targets)
        inputs_real = torch.cat([inputs_real_convs, z_tar, s_tar], dim=1)

        eps = torch.randn((targets.shape[0], self.total_latent_size)).to(self.device)
        inputs_fake = torch.cat([self.discriminator_target_convs(self.decode(eps)), eps], dim=1)

        total_inputs = torch.cat([inputs_real, inputs_fake], dim=0)
        total_labels = torch.cat(
            [torch.zeros(inputs_real.shape[0]),
             torch.ones(inputs_fake.shape[0])], dim=0).to(self.device).reshape(-1, 1)

        discriminator_outputs = F.sigmoid(self.discriminator_target(total_inputs))
        loss_a = F.binary_cross_entropy(input=discriminator_outputs, target=total_labels, reduction='sum')

        # Loss term "b"
        # Want to discriminate between "real" background data points and sampled ones

        inputs_real_convs = self.discriminator_reference_convs(background)
        inputs_real = torch.cat([inputs_real_convs, z_bg], dim=1)
        eps = torch.randn((background.shape[0], self.background_latent_size)).to(self.device)
        inputs_fake = torch.cat([
            self.discriminator_reference_convs(self.decode(
                torch.cat([eps, self.reference_vector.repeat(eps.shape[0]).reshape(eps.shape[0], -1).to(self.device)], dim=1))),
            eps], dim=1)

        total_inputs = torch.cat([inputs_real, inputs_fake], dim=0)
        total_labels = torch.cat(
            [torch.zeros(inputs_real.shape[0]),
             torch.ones(inputs_fake.shape[0])], dim=0).to(self.device).reshape(-1, 1)
        discriminator_outputs = F.sigmoid(self.discriminator_reference(total_inputs))

        loss_b = F.binary_cross_entropy(input=discriminator_outputs, target=total_labels, reduction='sum')

        # Loss term "c"
        # Reconstruction error for target samples

        loss_c = MSE_tar

        # Loss term "d"
        # Reconstruction error for background samples

        loss_d = MSE_bg

        # Loss term "e"
        # Latent variable reconstruction error for target samples
        eps = torch.randn((targets.shape[0], self.total_latent_size)).to(self.device)
        inputs_fake = self.decode(eps)
        _, _, _, _, _, z_tar, s_tar = self.forward_target(inputs_fake)
        eps_fake = torch.cat([z_tar, s_tar], dim=1)
        loss_e = F.mse_loss(eps, eps_fake, reduction='sum')

        # Loss term "f"
        # Latent variable reconstruction error for background samples
        eps = torch.randn((targets.shape[0], self.background_latent_size)).to(self.device)
        inputs_fake = self.decode(
            torch.cat([eps, self.reference_vector.repeat(eps.shape[0]).reshape(eps.shape[0], -1).to(self.device)], dim=1))
        _, _, _, _, _, z_bg, s_bg = self.forward_background(inputs_fake)
        eps_fake = z_bg
        loss_f = F.mse_loss(eps, eps_fake, reduction='sum')

        # Total loss
        loss = loss_a + loss_b + loss_c + loss_d + loss_e + loss_f
        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

        self.log('loss_a', loss_a, prog_bar=True)
        self.log('loss_b', loss_b, prog_bar=True)
        self.log('loss_c', loss_c, prog_bar=True)
        self.log('loss_d', loss_d, prog_bar=True)
        self.log('loss_e', loss_e, prog_bar=True)
        self.log('loss_f', loss_f, prog_bar=True)

        return loss

    def configure_optimizers(self):

        params = chain(
            self.z_convs.parameters(),
            self.z_mu.parameters(),
            self.z_var.parameters(),
            self.d_fc_1.parameters(),
            self.decode_convs.parameters(),
            self.discriminator_reference_convs.parameters(),
            self.discriminator_target_convs.parameters(),
            self.discriminator_target.parameters(),
            self.discriminator_reference.parameters(),
            [self.reference_vector]
        )

        opt = torch.optim.Adam(params)
        return opt
