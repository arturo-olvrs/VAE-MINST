import torch
import torch.nn as nn

class GaussianEncoder(nn.Module):
    """
    Encoder for a Gaussian Variational Autoencoder (VAE).
    Maps input data to the parameters of q_theta(z|x),
        which is the distribution of the latent variable z given the input x.

    Attributes:
    - hidden: Linear layer that maps input data to a hidden representation.
    - fc_mu: Linear layer that maps the hidden representation to the mean of q_theta(z|x).
    - fc_logvar: Linear layer that maps the hidden representation to the log variance of q_theta(z|x).

        Since the variance must be positive, we output log_var (the logarithm of the variance) instead of var directly, and then we can exponentiate it to get the variance.
    """

    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=10):
        super().__init__()
        self.hidden     = nn.Linear(input_dim, hidden_dim)
        self.hidden_activation = nn.ReLU()

        self.fc_mu      = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar  = nn.Linear(hidden_dim, latent_dim)

        self.fc_mu_activation     = nn.Identity()
        self.fc_logvar_activation = nn.Identity()
        
    def forward(self, x):
        """
        Forward pass through the encoder.

        Parameters:
        - x: Input batch to process. x.size(0) is the batch size
        Returns:
        - mu: The mean of q_theta(z|x)
        - log_var: The log variance of q_theta(z|x).
        """
        x_flattened = x.view(x.size(0), -1)  # Flatten the input to (batch_size, input_dim)

        h = self.hidden_activation(self.hidden(x_flattened))

        mu = self.fc_mu_activation(self.fc_mu(h))
        log_var = self.fc_logvar_activation(self.fc_logvar(h))

        return mu, log_var
    

class GaussianDecoder(nn.Module):
    """
    Decoder for a Gaussian Variational Autoencoder (VAE).
    Maps latent vectors back to the original data space.

    Attributes:
    - hidden: Linear layer that maps latent vectors to a hidden representation.
    - output: Linear layer that maps the hidden representation to the mean of the reconstructed input distribution (p_theta(x|z)).    
    """
    def __init__(self, latent_dim=10, hidden_dim=512, output_dim=784):
        super().__init__()
        self.hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Identity()
        
    def forward(self, z):
        """
        Forward pass through the decoder.
        
        Parameters:
        - z: Latent vector to decode. z.size(0) is the batch size
        Returns:
        - The mean of the reconstructed input distribution (p_theta(x|z)).
        """
        h = self.hidden_activation(self.hidden(z))
        return self.output_activation(self.output(h))

class BernoulliDecoder(nn.Module):
    """
    Decoder for a Bernoulli Variational Autoencoder (VAE).
    Maps latent vectors back to the original data space.

    Attributes:
    - hidden: Linear layer that maps latent vectors to a hidden representation.
    - output: Linear layer that maps the hidden representation to the probability p of the reconstructed input distribution (p_theta(x|z)).    
    """
    def __init__(self, latent_dim=10, hidden_dim=512, output_dim=784):
        super().__init__()
        self.hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, z):
        """
        Forward pass through the decoder.

        Parameters:
        - z: Latent vector to decode. z.size(0) is the batch size
        Returns:
        - The probability p of the reconstructed input distribution (p_theta(x|z)).
        """
        h = self.hidden_activation(self.hidden(z))
        return self.output_activation(self.output(h))
    

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        This allows us to backpropagate through the sampling process.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)     # In each dimension of the latent space, we sample from N(0,1) to get a random value. This is the "noise" we add to the mean to get a sample from the distribution.
            return mu + eps * std
        else:
            # In evaluation mode, we just return the mean (mu) as the latent vector to avoid randomness in the output.
            return mu

    def forward(self, x):
        """
        Forward pass through the VAE.

        Parameters:
        - x: Input data.

        Returns:
        - x_recon: The parameter of p_theta(x|z), which is the reconstructed input distribution.
        - mu: Mean of q_theta(z|x).
        - log_var: Log variance of q_theta(z|x).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar