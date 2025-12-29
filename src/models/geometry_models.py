"""Geometry-regularized autoencoder models"""
import torch
import torch.nn as nn
from typing import List, Dict
from .base_models import DeterministicAE, VAE
from ..diagnostics.jacobian_utils import sample_orthonormal_vectors, compute_jvp_decoder
from ..diagnostics.geometric_metrics import compute_k_volume, compute_edc_k


class GeometryRegularizedAE(DeterministicAE):
    """
    Autoencoder with geometry-preserving regularization.

    Adds:
    - k-volume floor regularization
    - Encoder-decoder consistency penalty
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        lambda_k_volume: float = 0.1,
        lambda_edc: float = 0.1,
        k_values: List[int] = [1, 2, 4],
        volume_floor_tau: float = -10.0,
        eps: float = 1e-6
    ):
        super().__init__(input_dim, latent_dim, hidden_dims, activation)

        self.lambda_k_volume = lambda_k_volume
        self.lambda_edc = lambda_edc
        self.k_values = k_values
        self.volume_floor_tau = volume_floor_tau
        self.eps = eps

    def compute_geometry_regularization(self, x):
        """Compute geometry regularization terms"""
        device = x.device
        batch_size, input_dim = x.shape

        total_vol_penalty = 0.0
        total_edc_penalty = 0.0
        n_terms = 0

        for k in self.k_values:
            if k > min(input_dim, self.encoder.network[-1].out_features):
                continue

            # Sample random orthonormal directions
            V_k = sample_orthonormal_vectors(input_dim, k, 'random', device=device)

            # Compute k-volume
            log_vol_k = compute_k_volume(self.encoder, x, V_k, self.eps)

            # Volume floor penalty: max(0, tau - log_vol_k)
            vol_penalty = torch.relu(self.volume_floor_tau - log_vol_k).mean()
            total_vol_penalty += vol_penalty

            # EDC penalty
            if self.lambda_edc > 0:
                edc = compute_edc_k(self.encoder, self.decoder, x, V_k)
                total_edc_penalty += edc.mean()

            n_terms += 1

        # Average over k values
        if n_terms > 0:
            total_vol_penalty /= n_terms
            total_edc_penalty /= n_terms

        return total_vol_penalty, total_edc_penalty

    def loss_function(self, x, x_recon, z):
        # Base reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')

        # Geometry regularization
        vol_penalty, edc_penalty = self.compute_geometry_regularization(x)

        # Total loss
        loss = recon_loss + \
               self.lambda_k_volume * vol_penalty + \
               self.lambda_edc * edc_penalty

        return {
            'loss': loss,
            'recon_loss': recon_loss.item(),
            'vol_penalty': vol_penalty.item(),
            'edc_penalty': edc_penalty.item()
        }


class GeometryRegularizedVAE(VAE):
    """
    VAE with geometry-preserving regularization on the mean map.

    Applies geometric regularization to the mean encoder mu(x).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        beta: float = 1.0,
        lambda_k_volume: float = 0.1,
        lambda_edc: float = 0.1,
        k_values: List[int] = [1, 2, 4],
        volume_floor_tau: float = -10.0,
        eps: float = 1e-6,
        use_mmd_posterior: bool = False,
        lambda_mmd: float = 0.0
    ):
        super().__init__(input_dim, latent_dim, hidden_dims, activation, beta)

        self.lambda_k_volume = lambda_k_volume
        self.lambda_edc = lambda_edc
        self.k_values = k_values
        self.volume_floor_tau = volume_floor_tau
        self.eps = eps
        self.use_mmd_posterior = use_mmd_posterior
        self.lambda_mmd = lambda_mmd

    def compute_mmd(self, z_sample, z_prior):
        """Compute MMD between aggregated posterior and prior"""
        # Simple RBF kernel MMD
        def rbf_kernel(x, y, sigma=1.0):
            x_size = x.size(0)
            y_size = y.size(0)

            x = x.unsqueeze(1)  # (n, 1, d)
            y = y.unsqueeze(0)  # (1, m, d)

            dist = torch.sum((x - y) ** 2, dim=2)
            return torch.exp(-dist / (2 * sigma ** 2))

        # Compute kernel matrices
        k_xx = rbf_kernel(z_sample, z_sample).mean()
        k_yy = rbf_kernel(z_prior, z_prior).mean()
        k_xy = rbf_kernel(z_sample, z_prior).mean()

        mmd = k_xx + k_yy - 2 * k_xy
        return mmd

    def compute_geometry_regularization(self, x, mu):
        """Compute geometry regularization on mean map"""
        device = x.device
        batch_size, input_dim = x.shape

        # Create a mean encoder network for Jacobian computation
        class MeanEncoder(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, x):
                h = self.encoder.shared(x)
                return self.encoder.fc_mu(h)

        mean_encoder = MeanEncoder(self.encoder)

        total_vol_penalty = 0.0
        total_edc_penalty = 0.0
        n_terms = 0

        for k in self.k_values:
            if k > min(input_dim, self.latent_dim):
                continue

            # Sample random orthonormal directions
            V_k = sample_orthonormal_vectors(input_dim, k, 'random', device=device)

            # Compute k-volume of mean map
            log_vol_k = compute_k_volume(mean_encoder, x, V_k, self.eps)

            # Volume floor penalty
            vol_penalty = torch.relu(self.volume_floor_tau - log_vol_k).mean()
            total_vol_penalty += vol_penalty

            # EDC penalty (using mean map and decoder)
            if self.lambda_edc > 0:
                edc = compute_edc_k(mean_encoder, self.decoder, x, V_k)
                total_edc_penalty += edc.mean()

            n_terms += 1

        if n_terms > 0:
            total_vol_penalty /= n_terms
            total_edc_penalty /= n_terms

        return total_vol_penalty, total_edc_penalty

    def loss_function(self, x, x_recon, mu, logvar, kl_weight=None):
        # Base VAE loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Geometry regularization on mean map
        vol_penalty, edc_penalty = self.compute_geometry_regularization(x, mu)

        # KL or MMD
        beta = kl_weight if kl_weight is not None else self.beta

        if self.use_mmd_posterior and self.lambda_mmd > 0:
            # Sample from posterior and prior for MMD
            z_sample = self.reparameterize(mu, logvar)
            z_prior = torch.randn_like(z_sample)
            mmd_loss = self.compute_mmd(z_sample, z_prior)

            loss = recon_loss + \
                   self.lambda_mmd * mmd_loss + \
                   self.lambda_k_volume * vol_penalty + \
                   self.lambda_edc * edc_penalty

            return {
                'loss': loss,
                'recon_loss': recon_loss.item(),
                'kl_div': kl_div.item(),
                'mmd_loss': mmd_loss.item(),
                'vol_penalty': vol_penalty.item(),
                'edc_penalty': edc_penalty.item(),
                'beta': beta
            }
        else:
            loss = recon_loss + \
                   beta * kl_div + \
                   self.lambda_k_volume * vol_penalty + \
                   self.lambda_edc * edc_penalty

            return {
                'loss': loss,
                'recon_loss': recon_loss.item(),
                'kl_div': kl_div.item(),
                'vol_penalty': vol_penalty.item(),
                'edc_penalty': edc_penalty.item(),
                'beta': beta
            }


# ============================================================================
# ABLATION VARIANTS: Coverage-Enhanced GA-AE
# ============================================================================

class GA_AE_Ablation(DeterministicAE):
    """
    GA-AE with ablation study for coverage terms.

    Variants:
    - add_ed: Energy Distance to real data
    - add_repulsion: Repulsion term for latent diversity
    - add_gap: Gap matching (on/off-manifold decoder behavior)
    - add_dec_vol: Decoder volume floor
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        # Geometry terms (original)
        lambda_k_volume: float = 0.1,
        lambda_edc: float = 0.1,
        k_values: List[int] = [1, 2, 4],
        volume_floor_tau: float = -10.0,
        eps: float = 1e-6,
        # Coverage terms (ablation)
        add_ed: bool = False,
        lambda_ed: float = 0.05,
        add_repulsion: bool = False,
        lambda_repel: float = 0.01,
        repulsion_sigma: float = 1.0,
        add_gap: bool = False,
        lambda_gap: float = 0.5,
        add_dec_vol: bool = False,
        lambda_dec_vol: float = 0.1,
        dec_vol_tau: float = -5.0
    ):
        super().__init__(input_dim, latent_dim, hidden_dims, activation)

        # Original geometry params
        self.lambda_k_volume = lambda_k_volume
        self.lambda_edc = lambda_edc
        self.k_values = k_values
        self.volume_floor_tau = volume_floor_tau
        self.eps = eps

        # Ablation params
        self.add_ed = add_ed
        self.lambda_ed = lambda_ed
        self.add_repulsion = add_repulsion
        self.lambda_repel = lambda_repel
        self.repulsion_sigma = repulsion_sigma
        self.add_gap = add_gap
        self.lambda_gap = lambda_gap
        self.add_dec_vol = add_dec_vol
        self.lambda_dec_vol = lambda_dec_vol
        self.dec_vol_tau = dec_vol_tau

        # Cache for real data (for ED computation)
        self.register_buffer('real_data_cache', None)
        self.cache_size = 2000

    def compute_geometry_regularization(self, x):
        """Compute original geometry regularization terms"""
        device = x.device
        batch_size, input_dim = x.shape

        total_vol_penalty = 0.0
        total_edc_penalty = 0.0
        n_terms = 0

        for k in self.k_values:
            if k > min(input_dim, self.encoder.network[-1].out_features):
                continue

            # Sample random orthonormal directions
            V_k = sample_orthonormal_vectors(input_dim, k, 'random', device=device)

            # Compute k-volume
            log_vol_k = compute_k_volume(self.encoder, x, V_k, self.eps)

            # Volume floor penalty: max(0, tau - log_vol_k)
            vol_penalty = torch.relu(self.volume_floor_tau - log_vol_k).mean()
            total_vol_penalty += vol_penalty

            # EDC penalty
            if self.lambda_edc > 0:
                edc = compute_edc_k(self.encoder, self.decoder, x, V_k)
                total_edc_penalty += edc.mean()

            n_terms += 1

        # Average over k values
        if n_terms > 0:
            total_vol_penalty /= n_terms
            total_edc_penalty /= n_terms

        return total_vol_penalty, total_edc_penalty

    def compute_energy_distance_loss(self, x, z_prior):
        """
        Compute differentiable Energy Distance between real data and generated data.

        ED = 2 E[||X - G(Z)||] - E[||X - X'||] - E[||G(Z) - G(Z')||]
        """
        # Generate samples from prior
        x_gen = self.decoder(z_prior)

        # Pairwise distances
        # ||X - G(Z)||
        x_expanded = x.unsqueeze(1)  # (batch, 1, dim)
        x_gen_expanded = x_gen.unsqueeze(0)  # (1, batch, dim)
        dist_cross = torch.sqrt(torch.sum((x_expanded - x_gen_expanded) ** 2, dim=2) + 1e-8)

        # ||X - X'||
        x_expanded2 = x.unsqueeze(0)  # (1, batch, dim)
        dist_xx = torch.sqrt(torch.sum((x_expanded - x_expanded2) ** 2, dim=2) + 1e-8)

        # ||G(Z) - G(Z')||
        x_gen_expanded2 = x_gen.unsqueeze(0)  # (1, batch, dim)
        dist_gen_gen = torch.sqrt(torch.sum((x_gen_expanded - x_gen_expanded2) ** 2, dim=2) + 1e-8)

        # Energy distance
        ed = 2 * dist_cross.mean() - dist_xx.mean() - dist_gen_gen.mean()
        return ed

    def compute_repulsion_loss(self, z):
        """
        Compute repulsion loss to encourage diversity in latent space.

        Loss = -log(mean(exp(-||z_i - z_j||^2 / sigma^2)))
        """
        # Pairwise distances
        z_expanded = z.unsqueeze(1)  # (batch, 1, latent_dim)
        z_expanded2 = z.unsqueeze(0)  # (1, batch, latent_dim)

        dist_sq = torch.sum((z_expanded - z_expanded2) ** 2, dim=2)

        # Gaussian kernel (exclude diagonal)
        batch_size = z.size(0)
        mask = 1.0 - torch.eye(batch_size, device=z.device)

        kernel = torch.exp(-dist_sq / (2 * self.repulsion_sigma ** 2))
        kernel = kernel * mask

        # Repulsion loss: encourage small kernel values (large distances)
        # Use -log(mean(kernel)) to penalize close points
        repulsion = -torch.log(kernel.sum() / (batch_size * (batch_size - 1)) + 1e-8)
        return repulsion

    def compute_decoder_wedge_volume(self, z):
        """
        Compute decoder wedge volume: s(z) = sqrt(det(J_D(z)^T J_D(z)))
        Returns log s(z) for numerical stability.
        """
        batch_size, latent_dim = z.shape

        # Sample latent_dim orthonormal directions
        U_k = sample_orthonormal_vectors(latent_dim, latent_dim, 'random', device=z.device)

        # Compute J_D(z) @ U_k
        B_k = compute_jvp_decoder(self.decoder, z, U_k)  # (batch, output_dim, latent_dim)

        # Gram matrix: B_k^T @ B_k
        gram = torch.bmm(B_k.transpose(1, 2), B_k)  # (batch, latent_dim, latent_dim)

        # log wedge volume = 0.5 * log det(Gram + eps I)
        eye = torch.eye(latent_dim, device=z.device).unsqueeze(0)
        gram_reg = gram + self.eps * eye

        # Compute log determinant
        log_det = torch.logdet(gram_reg + 1e-8)
        log_wedge_vol = 0.5 * log_det

        return log_wedge_vol

    def compute_gap_matching_loss(self, x, z_post, z_prior):
        """
        Compute gap matching: |E[log s(z_prior)] - E[log s(z_post)]|

        Matches on-manifold and off-manifold decoder behavior.
        """
        # Compute decoder wedge volumes
        log_s_post = self.compute_decoder_wedge_volume(z_post)
        log_s_prior = self.compute_decoder_wedge_volume(z_prior)

        # Gap: absolute difference of means
        gap = torch.abs(log_s_post.mean() - log_s_prior.mean())
        return gap

    def compute_decoder_volume_floor_loss(self, z):
        """
        Compute decoder volume floor penalty: ReLU(tau_dec - log s(z))
        """
        log_s = self.compute_decoder_wedge_volume(z)
        vol_floor_penalty = torch.relu(self.dec_vol_tau - log_s).mean()
        return vol_floor_penalty

    def loss_function(self, x, x_recon, z):
        # Base reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')

        # Original geometry regularization
        vol_penalty, edc_penalty = self.compute_geometry_regularization(x)

        # Initialize total loss
        loss = recon_loss + \
               self.lambda_k_volume * vol_penalty + \
               self.lambda_edc * edc_penalty

        loss_dict = {
            'loss': loss,
            'recon_loss': recon_loss.item(),
            'vol_penalty': vol_penalty.item(),
            'edc_penalty': edc_penalty.item()
        }

        # Coverage terms (ablation)

        # 1. Energy Distance
        if self.add_ed:
            # Sample from prior
            batch_size = x.size(0)
            z_prior = torch.randn(batch_size, z.size(1), device=x.device)

            ed_loss = self.compute_energy_distance_loss(x, z_prior)
            loss = loss + self.lambda_ed * ed_loss
            loss_dict['ed_loss'] = ed_loss.item()

        # 2. Repulsion
        if self.add_repulsion:
            repulsion_loss = self.compute_repulsion_loss(z)
            loss = loss + self.lambda_repel * repulsion_loss
            loss_dict['repulsion_loss'] = repulsion_loss.item()

        # 3. Gap Matching
        if self.add_gap:
            batch_size = x.size(0)
            z_prior = torch.randn(batch_size, z.size(1), device=x.device)

            gap_loss = self.compute_gap_matching_loss(x, z, z_prior)
            loss = loss + self.lambda_gap * gap_loss
            loss_dict['gap_loss'] = gap_loss.item()

        # 4. Decoder Volume Floor
        if self.add_dec_vol:
            dec_vol_loss = self.compute_decoder_volume_floor_loss(z)
            loss = loss + self.lambda_dec_vol * dec_vol_loss
            loss_dict['dec_vol_loss'] = dec_vol_loss.item()

        # Update loss in dict
        loss_dict['loss'] = loss

        return loss_dict


# ============================================================================
# GA-NATIVE REGULARIZERS: Grassmann Spread, Blade Entropy, Blade Matching
# ============================================================================

class GA_AE_Grassmann(DeterministicAE):
    """
    GA-AE with true geometric algebra / exterior algebra regularizers.

    Implements:
    1. Grassmann spread: Repel tangent k-blades (not just latent positions)
    2. Blade entropy: Maximize entropy across k-volume grades
    3. Blade-moment matching: MMD on blade spectrum between on/off-manifold
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        # Original geometry terms
        lambda_k_volume: float = 0.1,
        lambda_edc: float = 0.1,
        k_values: List[int] = [1, 2],
        volume_floor_tau: float = -10.0,
        eps: float = 1e-6,
        # GA-native terms
        add_grassmann: bool = False,
        lambda_grassmann: float = 0.1,
        add_blade_entropy: bool = False,
        lambda_blade_entropy: float = 0.1,
        add_blade_matching: bool = False,
        lambda_blade_matching: float = 0.1,
        blade_mmd_sigma: float = 1.0
    ):
        super().__init__(input_dim, latent_dim, hidden_dims, activation)

        # Original geometry params
        self.lambda_k_volume = lambda_k_volume
        self.lambda_edc = lambda_edc
        self.k_values = k_values
        self.volume_floor_tau = volume_floor_tau
        self.eps = eps

        # GA-native params
        self.add_grassmann = add_grassmann
        self.lambda_grassmann = lambda_grassmann
        self.add_blade_entropy = add_blade_entropy
        self.lambda_blade_entropy = lambda_blade_entropy
        self.add_blade_matching = add_blade_matching
        self.lambda_blade_matching = lambda_blade_matching
        self.blade_mmd_sigma = blade_mmd_sigma

    def compute_geometry_regularization(self, x):
        """Compute original geometry regularization terms"""
        device = x.device
        batch_size, input_dim = x.shape

        total_vol_penalty = 0.0
        total_edc_penalty = 0.0
        n_terms = 0

        for k in self.k_values:
            if k > min(input_dim, self.encoder.network[-1].out_features):
                continue

            # Sample random orthonormal directions
            V_k = sample_orthonormal_vectors(input_dim, k, 'random', device=device)

            # Compute k-volume
            log_vol_k = compute_k_volume(self.encoder, x, V_k, self.eps)

            # Volume floor penalty: max(0, tau - log_vol_k)
            vol_penalty = torch.relu(self.volume_floor_tau - log_vol_k).mean()
            total_vol_penalty += vol_penalty

            # EDC penalty
            if self.lambda_edc > 0:
                edc = compute_edc_k(self.encoder, self.decoder, x, V_k)
                total_edc_penalty += edc.mean()

            n_terms += 1

        # Average over k values
        if n_terms > 0:
            total_vol_penalty /= n_terms
            total_edc_penalty /= n_terms

        return total_vol_penalty, total_edc_penalty

    def compute_decoder_k_blades(self, z, k):
        """
        Compute decoder k-blades: B_k(z) = J_D(z) @ V_k
        Returns: (batch, output_dim, k) matrices representing k-blades
        """
        batch_size, latent_dim = z.shape

        # Sample k orthonormal directions in latent space
        V_k = sample_orthonormal_vectors(latent_dim, k, 'random', device=z.device)

        # Compute J_D(z) @ V_k via JVP
        B_k = compute_jvp_decoder(self.decoder, z, V_k)  # (batch, output_dim, k)

        return B_k

    def compute_blade_volumes(self, z, k_values=None):
        """
        Compute blade volumes s_k(z) = |∧^k J_D(z)| for multiple k.
        Returns dict {k: log_s_k} where log_s_k has shape (batch,)
        """
        if k_values is None:
            k_values = self.k_values

        blade_vols = {}

        for k in k_values:
            if k > z.shape[1]:  # k <= latent_dim
                continue

            # Get k-blade
            B_k = self.compute_decoder_k_blades(z, k)  # (batch, output_dim, k)

            # Compute volume: sqrt(det(B_k^T B_k))
            gram = torch.bmm(B_k.transpose(1, 2), B_k)  # (batch, k, k)

            # Regularize and compute log determinant
            eye = torch.eye(k, device=z.device).unsqueeze(0)
            gram_reg = gram + self.eps * eye

            log_det = torch.logdet(gram_reg + 1e-8)
            log_vol = 0.5 * log_det

            blade_vols[k] = log_vol

        return blade_vols

    def compute_grassmann_spread_loss(self, z, n_pairs=32):
        """
        Grassmann spread regularizer: Repel tangent k-blades.

        For k-blades B_i, B_j, compute similarity:
        sim(B_i, B_j) = sqrt(det((U_i^T U_j)(U_j^T U_i))) / sqrt(det(U_i^T U_i) det(U_j^T U_j))

        where U_i = J_D(z_i) V_k.

        This is the Grassmann distance (principal angle overlap).
        We want to MINIMIZE similarity → maximize diversity.
        """
        batch_size = z.shape[0]

        # Sample pairs
        n_pairs = min(n_pairs, batch_size // 2)
        if n_pairs < 2:
            return torch.tensor(0.0, device=z.device)

        # Randomly sample pairs
        idx = torch.randperm(batch_size, device=z.device)[:2*n_pairs]
        idx_i = idx[:n_pairs]
        idx_j = idx[n_pairs:2*n_pairs]

        z_i = z[idx_i]
        z_j = z[idx_j]

        total_sim = 0.0
        n_terms = 0

        # Compute for each k
        for k in self.k_values:
            if k > z.shape[1]:
                continue

            # Get k-blades for sampled pairs
            B_i = self.compute_decoder_k_blades(z_i, k)  # (n_pairs, output_dim, k)
            B_j = self.compute_decoder_k_blades(z_j, k)  # (n_pairs, output_dim, k)

            # Gram matrices
            G_i = torch.bmm(B_i.transpose(1, 2), B_i)  # (n_pairs, k, k)
            G_j = torch.bmm(B_j.transpose(1, 2), B_j)  # (n_pairs, k, k)

            # Cross Gram
            G_ij = torch.bmm(B_i.transpose(1, 2), B_j)  # (n_pairs, k, k)
            G_ji = G_ij.transpose(1, 2)  # (n_pairs, k, k)

            # Grassmann similarity: sqrt(det(G_ij @ G_ji)) / sqrt(det(G_i) det(G_j))
            # Regularize for numerical stability
            eye = torch.eye(k, device=z.device).unsqueeze(0)
            eps = self.eps

            det_i = torch.det(G_i + eps * eye).clamp(min=1e-8)
            det_j = torch.det(G_j + eps * eye).clamp(min=1e-8)
            det_ij_ji = torch.det(torch.bmm(G_ij, G_ji) + eps * eye).clamp(min=1e-8)

            # Similarity (want to minimize this)
            sim = torch.sqrt(det_ij_ji) / (torch.sqrt(det_i * det_j) + 1e-8)

            # Average similarity for this k
            total_sim += sim.mean()
            n_terms += 1

        # Return average similarity across k values
        if n_terms > 0:
            return total_sim / n_terms
        else:
            return torch.tensor(0.0, device=z.device)

    def compute_blade_entropy_loss(self, z):
        """
        Blade entropy regularizer: Maximize entropy across k-volume grades.

        For each z, compute blade spectrum:
        p_k(z) = s_k(z) / (Σ_j s_j(z) + δ)

        Then maximize entropy: -Σ p_k log p_k
        (minimize negative entropy)
        """
        # Get blade volumes for all k
        blade_vols = self.compute_blade_volumes(z)  # dict {k: log_s_k}

        if len(blade_vols) == 0:
            return torch.tensor(0.0, device=z.device)

        # Convert log volumes to volumes: s_k = exp(log_s_k)
        # Stack into tensor: (batch, n_grades)
        s_k_list = []
        for k in sorted(blade_vols.keys()):
            log_s_k = blade_vols[k]
            s_k = torch.exp(log_s_k.clamp(max=10.0))  # clamp for stability
            s_k_list.append(s_k)

        s_k_tensor = torch.stack(s_k_list, dim=1)  # (batch, n_grades)

        # Normalize into probability distribution
        p_k = s_k_tensor / (s_k_tensor.sum(dim=1, keepdim=True) + self.eps)

        # Compute entropy: H = -Σ p_k log(p_k)
        log_p_k = torch.log(p_k + self.eps)
        entropy = -(p_k * log_p_k).sum(dim=1)  # (batch,)

        # We want to maximize entropy, so minimize negative entropy
        neg_entropy = -entropy.mean()

        return neg_entropy

    def compute_blade_matching_loss(self, z_post, z_prior):
        """
        Blade-moment matching: MMD between blade spectra.

        Feature map: φ(z) = (log s_1(z), ..., log s_K(z))
        Compute MMD(φ(z_post), φ(z_prior)) using RBF kernel.
        """
        # Compute blade spectra
        blade_post = self.compute_blade_volumes(z_post)  # dict {k: log_s_k}
        blade_prior = self.compute_blade_volumes(z_prior)  # dict {k: log_s_k}

        if len(blade_post) == 0 or len(blade_prior) == 0:
            return torch.tensor(0.0, device=z_post.device)

        # Stack into feature vectors: (batch, n_grades)
        phi_post_list = [blade_post[k] for k in sorted(blade_post.keys())]
        phi_prior_list = [blade_prior[k] for k in sorted(blade_prior.keys())]

        phi_post = torch.stack(phi_post_list, dim=1)  # (batch_post, n_grades)
        phi_prior = torch.stack(phi_prior_list, dim=1)  # (batch_prior, n_grades)

        # Compute MMD with RBF kernel
        def rbf_kernel(x, y, sigma):
            """RBF kernel between feature vectors"""
            x_expanded = x.unsqueeze(1)  # (n, 1, d)
            y_expanded = y.unsqueeze(0)  # (1, m, d)
            dist_sq = torch.sum((x_expanded - y_expanded) ** 2, dim=2)
            return torch.exp(-dist_sq / (2 * sigma ** 2))

        k_xx = rbf_kernel(phi_post, phi_post, self.blade_mmd_sigma).mean()
        k_yy = rbf_kernel(phi_prior, phi_prior, self.blade_mmd_sigma).mean()
        k_xy = rbf_kernel(phi_post, phi_prior, self.blade_mmd_sigma).mean()

        mmd = k_xx + k_yy - 2 * k_xy

        return mmd

    def loss_function(self, x, x_recon, z):
        # Base reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')

        # Original geometry regularization
        vol_penalty, edc_penalty = self.compute_geometry_regularization(x)

        # Initialize total loss
        loss = recon_loss + \
               self.lambda_k_volume * vol_penalty + \
               self.lambda_edc * edc_penalty

        loss_dict = {
            'loss': loss,
            'recon_loss': recon_loss.item(),
            'vol_penalty': vol_penalty.item(),
            'edc_penalty': edc_penalty.item()
        }

        # GA-native terms

        # 1. Grassmann spread
        if self.add_grassmann:
            grassmann_loss = self.compute_grassmann_spread_loss(z)
            loss = loss + self.lambda_grassmann * grassmann_loss
            loss_dict['grassmann_loss'] = grassmann_loss.item()

        # 2. Blade entropy
        if self.add_blade_entropy:
            blade_entropy_loss = self.compute_blade_entropy_loss(z)
            loss = loss + self.lambda_blade_entropy * blade_entropy_loss
            loss_dict['blade_entropy_loss'] = blade_entropy_loss.item()

        # 3. Blade matching (on/off-manifold)
        if self.add_blade_matching:
            batch_size = x.size(0)
            z_prior = torch.randn(batch_size, z.size(1), device=x.device)

            blade_matching_loss = self.compute_blade_matching_loss(z, z_prior)
            loss = loss + self.lambda_blade_matching * blade_matching_loss
            loss_dict['blade_matching_loss'] = blade_matching_loss.item()

        # Update loss in dict
        loss_dict['loss'] = loss

        return loss_dict
