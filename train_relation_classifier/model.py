import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


# Transformer-based Relation Classifier
class RelationClassifierTransformer(nn.Module):
    def __init__(self, rel_name_to_idx, idx_to_rel_name, num_timesteps, map_size, d_model, classifier_config):
        super(RelationClassifierTransformer, self).__init__()
        self.rel_name_to_idx = rel_name_to_idx
        self.idx_to_rel_name = idx_to_rel_name
        self.num_timesteps = num_timesteps
        self.map_size = map_size
        self.d_model = d_model
        self.classifier_config = classifier_config
        self.mode = self.classifier_config["mode"]
        num_relations = len(self.rel_name_to_idx.keys())
        dropout_percent = self.classifier_config["dropout"]
        
        self.num_input_maps = 2

        # Timestep embedding
        self.timestep_embedding_dim = self.classifier_config["timestep_embedding_dim"]
        if self.timestep_embedding_dim > 0:
            self.timestep_embedding = nn.Embedding(num_timesteps, self.timestep_embedding_dim)
            
        self.positional_encoding = PositionalEncoding(d_model)

        # Input projection to match d_model for attention layers
        self.projection = nn.Linear(self.map_size * self.map_size * self.num_input_maps, d_model)
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=self.classifier_config["nhead"], 
            dim_feedforward=self.classifier_config["dim_feedforward"], 
            dropout=dropout_percent, 
            activation="relu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=self.classifier_config["num_layers"])

        # Fully connected layers with layer normalization
        self.fc = nn.Sequential(
            nn.Linear(d_model + self.timestep_embedding_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_percent),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_percent),
            nn.Linear(128, num_relations)
        )

    
    def forward(self, maps, timestep):
        x = maps
        x = x.view(x.size(0), -1)  # shape = (batch_size, 2 * 16 * 16)
        x = self.projection(x)  # shape = (batch_size, d_model)
        x = self.positional_encoding(x.unsqueeze(1))  # shape = (batch_size, 1, d_model)
        x = self.transformer(x)  # shape = (batch_size, 1, d_model)
        x = x.squeeze(1)  # shape = (batch_size, d_model)

        # Embed timestep and concatenate with features
        if self.timestep_embedding_dim > 0:
            timestep_embedding = self.timestep_embedding(timestep.squeeze(dim=1).long())  # shape = (batch_size, timestep_embedding_dim)
            x = torch.cat((x, timestep_embedding), dim=1)  # shape = (batch_size, d_model + timestep_embedding_dim)
        
        x = self.fc(x)
        return x


class CrossAttentionExtractor(nn.Module):
    def __init__(self, input_channels, map_size, d_model, num_timesteps, extractor_config):
        super(CrossAttentionExtractor, self).__init__()
        self.input_channels = input_channels
        self.map_size = map_size
        self.d_model = d_model
        self.extractor_config = extractor_config
        self.use_mean = self.extractor_config.get("use_mean", False)

        dropout_percent = self.extractor_config["dropout"] if "dropout" in self.extractor_config else 0.1
        
        if self.use_mean:
            return
        
        # Flatten input spatial dimensions (16x16) into a single vector
        self.input_projection = nn.Linear(map_size * map_size, d_model)
        
        # Timestep embedding
        if self.extractor_config["timestep_embedding"]["enable"]:
            self.timestep_embedding = nn.Embedding(num_timesteps, d_model)

        # Positional encoding for the transformer
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=input_channels)
        
        # Lightweight transformer for layer aggregation
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=self.extractor_config["nhead"], 
            dim_feedforward=self.extractor_config["dim_feedforward"],
            dropout=dropout_percent, 
            activation="relu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.extractor_config["num_layers"]
        )

        # Learnable query vector for attention-based aggregation
        self.query_vector = nn.Parameter(torch.randn(1, 1, d_model))  # shape = (1, 1, d_model)
        
        self.layer_attention = nn.MultiheadAttention(embed_dim=d_model, 
                                                     num_heads=self.extractor_config["layer_attn_nheads"], 
                                                     batch_first=True)

        # Output projection back to spatial dimensions (16x16)
        self.output_projection = nn.Linear(d_model, map_size * map_size)


    def forward(self, cross_attention_map, timestep):
        """
        Input: cross_attention_map (B, 57, 16, 16)
        Output: aggregated_map (B, 16, 16)
        """
        
        if self.use_mean:
            x = cross_attention_map.mean(dim=1)
            return x
        
        # Flatten spatial dimensions and project to d_model
        b, c, h, w = cross_attention_map.shape
        x = cross_attention_map.view(b, c, -1)  # shape = (B, 57 * 24, 256)
        x = self.input_projection(x)  # shape = (B, 57 * 24, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)  # shape = (B, 57 * 24, d_model)

        # Inject timestep embedding:
        apply_transformer = True
        if self.extractor_config["timestep_embedding"]["enable"]:
            timestep_emb = self.timestep_embedding(timestep.squeeze(dim=1).long()).unsqueeze(1)  # shape = (B, 1, d_model)
            mode = self.extractor_config["timestep_embedding"]["mode"]
            if mode == "pre":
                x = x + timestep_emb  # shape = (B, 57, d_model)

            elif mode == "mid":
                apply_transformer = False
                for layer in self.transformer.layers:
                    x = layer(x + timestep_emb)
        
        if apply_transformer:
            x = self.transformer(x)  # shape = (B, 57, d_model)

        # Attention-based aggregation using learnable query
        query = self.query_vector.repeat(x.shape[0], 1, 1)  # shape = (B, 1, d_model)
        attn_output, _ = self.layer_attention(query, x, x)  # shape = (B, 1, d_model)
        attn_output = attn_output.squeeze(1)  # shape = (B, d_model)

        # Project back to spatial dimensions
        aggregated_map = self.output_projection(attn_output)  # shape = (B, 256)
        aggregated_map = aggregated_map.view(b, h, w)  # shape = (B, 16, 16)

        return aggregated_map


class EnhancedRelationClassifier(nn.Module):
    def __init__(self, rel_name_to_idx, idx_to_rel_name, input_channels, config, training_config=None):
        super(EnhancedRelationClassifier, self).__init__()
        self.rel_name_to_idx = rel_name_to_idx
        self.idx_to_rel_name = idx_to_rel_name
        self.input_channels = input_channels
        self.config = config
        self.training_config = training_config
        self.num_timesteps = self.config["num_timesteps"]
        self.map_size = self.config["map_size"]
        self.d_model = self.config["d_model"]
        
        self.timestep_emb_config = config["extractor"]["timestep_embedding"]
        self.timestep_embedding_mode = self.timestep_emb_config["mode"] if self.timestep_emb_config["enable"] else ""
        
        self.object_extractor = CrossAttentionExtractor(
            input_channels=input_channels, 
            map_size=self.map_size,
            d_model=self.d_model,
            num_timesteps=self.num_timesteps,
            extractor_config=self.config["extractor"],
        )
    
        self.relation_classifier = RelationClassifierTransformer(self.rel_name_to_idx, self.idx_to_rel_name, 
                                                                 self.num_timesteps, self.map_size, self.d_model,
                                                                 self.config["classifier"])

    def forward(self, maps, timestep):
        # Input maps: (B, num_maps, C, H, W)
        # Handle cross-attn maps:
        map1 = maps[:, 0]
        x1 = self.object_extractor(map1, timestep)
        map2 = maps[:, 1]
        x2 = self.object_extractor(map2, timestep)
        extracted_maps = torch.stack([x1, x2], dim=1)
        output = self.relation_classifier(extracted_maps, timestep)
        return output
    
    def relation_name_to_index(self, rel_name):
        return self.rel_name_to_idx[rel_name] if rel_name in self.rel_name_to_idx else -1
    
    def index_to_relation_name(self, idx):
        return self.idx_to_rel_name[idx] if idx in self.idx_to_rel_name else ""
    