import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=196, hidden_dim=128, latent_dim=8):
        super(VAE, self).__init__()

        # -----------
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.training = True
        
        # --- ENCODER ---
        """
        The encoder maps the input data to a latent space.
        Given an input x, it outputs the parameters of the latent distribution (q_theta(z|x)).
            Since we use a Gaussian VAE, it outputs the mean (mu) and variance (var) of the Gaussian distribution.
            Since the variance must be positive, we output log_var (the logarithm of the variance) instead of var directly, and then we can exponentiate it to get the variance.

        It consists of two linear layers:
        1. encoder_hidden: Maps the input to a hidden representation.
        2.1. fc_mu: Maps the hidden representation to the mean of the latent distribution.
        2.2. fc_logvar: Maps the hidden representation to the log variance of the latent distribution.
        """

        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        self.encoder_hidden_activation = nn.Tanh()
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_mu_activation = nn.Identity()
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) 
        self.fc_logvar_activation = nn.Identity()

        # --- DECODER ---
        """
        The decoder maps the latent space back to the original data space.
        Given a latent vector z, it outputs the parameters of the reconstructed input distribution (p_theta(x|z)).
            Since we use a Gaussian VAE, it outputs the mean of the mean (mu) of the Gaussian distribution.
            The variance is assumed to be fixed (e.g., 1) for simplicity.
        
        It consists of two linear layers:
        1. decoder_hidden: Maps the latent vector z to a hidden representation.
        2. decoder_output: Maps the hidden representation to the mean of the reconstructed input distribution.
        """
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_hidden_activation = nn.Tanh()
        self.decoder_output = nn.Linear(hidden_dim, input_dim)
        self.decoder_output_activation = nn.Identity()

        
        
    def encode(self, x):
        """
        Encode the input data x into the latent space.

        Returns the mean (mu) and log variance (log_var) of the latent distribution.
        """
        hidden = self.encoder_hidden_activation(self.encoder_hidden(x))

        mu      = self.fc_mu_activation     (self.fc_mu(hidden))
        log_var = self.fc_logvar_activation (self.fc_logvar(hidden))
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).

        This allows us to backpropagate through the sampling process.
        """
        # // TODO: Different
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)     # In each dimension of the latent space, we sample from N(0,1) to get a random value. This is the "noise" we add to the mean to get a sample from the distribution.
            return mu + eps * std
        else:
            # In evaluation mode, we just return the mean (mu) as the latent vector to avoid randomness in the output.
            return mu

    def decode(self, z):
        """Reconstruct the input data from the latent vector z."""
        hidden = self.decoder_hidden_activation(self.decoder_hidden(z))
        return self.decoder_output_activation(self.decoder_output(hidden)) 
    
    def forward(self, x):
        """
        Forward pass through the VAE.

        Parameters:
        - x: Input batch to process. x.size(0) is the batch size

        Returns:
        - mu_reconstruction: The mean of the reconstructed input distribution (p_theta(x|z)).
        - mu_latent: The mean of the latent distribution (q_theta(z|x)).
        - log_var_latent: The log variance of the latent distribution (q_theta(z|x)).
        """
        
        # Flattens the input automatically to the expected dimension (batch_size, input_dim)
        x_flat = x.view(x.size(0), -1)

        mu_latent, log_var_latent = self.encode(x_flat)      # They are the parameters of the latent distribution (q_theta(z|x))
        z = self.reparameterize(mu_latent, log_var_latent)

        mu_reconstruction = self.decode(z)
        return mu_reconstruction, mu_latent, log_var_latent