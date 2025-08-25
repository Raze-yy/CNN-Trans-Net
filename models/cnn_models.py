# models/cnn_models.py

# models/cnn_models.py

import torch
import torch.nn as nn

# ============================================================
# 1. Basic CNN Model
# ============================================================

class ExcitationSpectrumModel_CNN(nn.Module):
    """
    Basic CNN model for excitation spectrum prediction.
    """

    def __init__(self, input_dim=12, cnn_hidden_dim=128, feature_dim=64, output_dim=41 * 49, kernel_size=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(cnn_hidden_dim * 49, feature_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        batch_size = x.size(0)
        source = x[:, :, [1, 3, 5, 7, 9, 11]]
        emission = x[:, :, [2, 4, 6, 8, 10, 12]]
        spectra = torch.cat([source, emission], dim=-1)
        spectra = spectra.permute(0, 2, 1)
        features = self.cnn(spectra)
        features = features.view(batch_size, -1)
        global_features = self.fc(features)
        output = self.decoder(global_features)
        return output.view(batch_size, 41, 49)

# ============================================================
# 2. Improved CNN + Transformer Model
# ============================================================

class ExcitationSpectrumModel_CNN_Transformer(nn.Module):
    """
    CNN + Transformer model with improved structure.
    """

    def __init__(self, cnn_hidden_dim=128, transformer_dim=128, num_heads=2, num_layers=1, output_dim=41 * 49, kernel_size=3):
        super().__init__()
        self.output_dim = output_dim

        self.source_fc = nn.Sequential(nn.Linear(6, cnn_hidden_dim), nn.ReLU())
        self.sample_fc = nn.Sequential(nn.Linear(6, cnn_hidden_dim), nn.ReLU())

        self.cnn = nn.Sequential(
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=num_heads, dim_feedforward=256, dropout=0.1),
            num_layers=num_layers
        )

        self.positional_encoding = nn.Parameter(torch.randn(1, 49, transformer_dim))

        self.cnn_final = nn.Sequential(
            nn.Conv1d(transformer_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(cnn_hidden_dim * 49, 128),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        batch_size = x.size(0)
        source = x[:, :, [1, 3, 5, 7, 9, 11]]
        emission = x[:, :, [2, 4, 6, 8, 10, 12]]

        source_embed = self.source_fc(source)
        sample_embed = self.sample_fc(emission)

        interaction = source_embed * sample_embed
        interaction = interaction.permute(0, 2, 1)
        cnn_out = self.cnn(interaction)

        cnn_out = cnn_out.permute(0, 2, 1) + self.positional_encoding
        trans_out = self.transformer_encoder(cnn_out)

        trans_out = trans_out.permute(0, 2, 1)
        cnn_out_final = self.cnn_final(trans_out)

        cnn_out_final = cnn_out_final.view(batch_size, -1)
        global_features = self.fc(cnn_out_final)
        output = self.decoder(global_features)
        return output.view(batch_size, 41, 49)

# ============================================================
# 3. CNN with Interaction Mechanism
# ============================================================

class ExcitationSpectrumModel_CNN_Interaction(nn.Module):
    """
    CNN model with source-sample interaction mechanism.
    """

    def __init__(self, cnn_hidden_dim=128, feature_dim=64, output_dim=41 * 49, kernel_size=5):
        super().__init__()
        self.source_fc = nn.Sequential(nn.Linear(6, cnn_hidden_dim), nn.ReLU())
        self.sample_fc = nn.Sequential(nn.Linear(6, cnn_hidden_dim), nn.ReLU())

        self.cnn = nn.Sequential(
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(cnn_hidden_dim * 49, feature_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        batch_size = x.size(0)
        source = self.source_fc(x[:, :, [1, 3, 5, 7, 9, 11]])
        sample = self.sample_fc(x[:, :, [2, 4, 6, 8, 10, 12]])

        interaction = source * sample
        interaction = interaction.permute(0, 2, 1)

        cnn_out = self.cnn(interaction)
        flat = cnn_out.view(batch_size, -1)
        features = self.fc(flat)
        output = self.decoder(features)
        return output.view(batch_size, 41, 49)

# ============================================================
# 4. CNN with Residual Connections
# ============================================================

class ExcitationSpectrumModel_CNN_ResNet(nn.Module):
    """
    CNN model with residual blocks.
    """

    def __init__(self, cnn_hidden_dim=128, feature_dim=64, output_dim=41 * 49, kernel_size=5):
        super().__init__()
        self.source_fc = nn.Sequential(nn.Linear(6, cnn_hidden_dim), nn.ReLU())
        self.sample_fc = nn.Sequential(nn.Linear(6, cnn_hidden_dim), nn.ReLU())

        self.block1 = nn.Sequential(
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2)
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size, padding=kernel_size // 2)
        )

        self.shortcut = nn.Conv1d(cnn_hidden_dim, cnn_hidden_dim, kernel_size=1)

        self.fc = nn.Sequential(
            nn.Linear(cnn_hidden_dim * 49, feature_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        batch_size = x.size(0)
        source = self.source_fc(x[:, :, [1, 3, 5, 7, 9, 11]])
        sample = self.sample_fc(x[:, :, [2, 4, 6, 8, 10, 12]])

        interaction = source * sample
        interaction = interaction.permute(0, 2, 1)

        out = self.block1(interaction) + self.shortcut(interaction)
        out = self.block2(out) + out

        flat = out.view(batch_size, -1)
        features = self.fc(flat)
        output = self.decoder(features)
        return output.view(batch_size, 41, 49)

# ============================================================
# 5. CNN with Multi-scale Convolution
# ============================================================

class ExcitationSpectrumModel_CNN_MultiScale(nn.Module):
    """
    CNN model with multi-scale convolution blocks.
    """

    def __init__(self, feature_dim=64, output_dim=41 * 49, kernel_size=5):
        super().__init__()
        self.source_fc = nn.Sequential(nn.Linear(6, 64), nn.ReLU())
        self.sample_fc = nn.Sequential(nn.Linear(6, 64), nn.ReLU())

        self.cnn = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size, padding=kernel_size // 2),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 49, feature_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        batch_size = x.size(0)
        source = self.source_fc(x[:, :, [1, 3, 5, 7, 9, 11]])
        sample = self.sample_fc(x[:, :, [2, 4, 6, 8, 10, 12]])

        interaction = source * sample
        interaction = interaction.permute(0, 2, 1)

        cnn_out = self.cnn(interaction)
        flat = cnn_out.view(batch_size, -1)
        features = self.fc(flat)
        output = self.decoder(features)
        return output.view(batch_size, 41, 49)

# ============================================================
# 6. CNN U-Net Style Model
# ============================================================

class ExcitationSpectrumModel_CNN_UNet(nn.Module):
    """
    U-Net like CNN model with skip connections.
    """

    def __init__(self, feature_dim=64, output_dim=41 * 49, kernel_size=5):
        super().__init__()
        self.source_fc = nn.Sequential(nn.Linear(6, 64), nn.ReLU())
        self.sample_fc = nn.Sequential(nn.Linear(6, 64), nn.ReLU())

        self.encoder1 = nn.Sequential(nn.Conv1d(64, 128, kernel_size, padding=kernel_size // 2), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Conv1d(128, 256, kernel_size, padding=kernel_size // 2), nn.ReLU())
        self.bottleneck = nn.Sequential(nn.Conv1d(256, 256, kernel_size, padding=kernel_size // 2), nn.ReLU())
        self.decoder1 = nn.Sequential(nn.Conv1d(256, 128, kernel_size, padding=kernel_size // 2), nn.ReLU())
        self.decoder2 = nn.Sequential(nn.Conv1d(128, 64, kernel_size, padding=kernel_size // 2), nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(64 * 49, feature_dim), nn.ReLU())
        self.decoder_final = nn.Sequential(nn.Linear(feature_dim, output_dim), nn.Softplus())

    def forward(self, x):
        batch_size = x.size(0)
        source = self.source_fc(x[:, :, [1, 3, 5, 7, 9, 11]])
        sample = self.sample_fc(x[:, :, [2, 4, 6, 8, 10, 12]])

        interaction = source * sample
        interaction = interaction.permute(0, 2, 1)

        enc1 = self.encoder1(interaction)
        enc2 = self.encoder2(enc1)
        bottleneck = self.bottleneck(enc2)
        dec1 = self.decoder1(bottleneck) + enc1
        dec2 = self.decoder2(dec1)

        flat = dec2.view(batch_size, -1)
        features = self.fc(flat)
        output = self.decoder_final(features)
        return output.view(batch_size, 41, 49)

# ============================================================
# Optional: Export all models
# ============================================================

__all__ = [
    "ExcitationSpectrumModel_CNN",
    "ExcitationSpectrumModel_CNN_Transformer",
    "ExcitationSpectrumModel_CNN_ResNet",
    "ExcitationSpectrumModel_CNN_UNet",
    "ExcitationSpectrumModel_CNN_MultiScale",
    "ExcitationSpectrumModel_CNN_Interaction"
]