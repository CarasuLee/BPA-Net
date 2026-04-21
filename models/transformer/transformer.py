from matplotlib import image
import math
import torch
import torch.nn.functional as F
from torch import nn
import copy
from models.containers import ModuleList
from models.transformer.utils import sinusoid_encoding_table
from models.beam_search import *
from ..captioning_model import CaptioningModel
from models.transformer.clip_backbone import CLIPTextEncoder


class TPA(nn.Module):
    def __init__(self, channels=1536, num_prototypes=32, temperature=0.1):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, channels)
        )
        
    def forward(self, x):
        b, c, seq_len = x.shape
        
        x_flat = x
        
        x_norm = F.normalize(x_flat, p=2, dim=1)
        
        proto_flat = self.prototypes
        
        x_norm_fp32 = x_norm.float()
        prototypes_fp32 = proto_flat.float()

        sim = torch.einsum('bcs, pc -> bps', x_norm_fp32, prototypes_fp32)
        
        weights = F.softmax(sim / self.temperature, dim=1)

        weights = weights.to(x_flat.dtype)
        
        aggregated = torch.bmm(weights, x_flat.transpose(1, 2))
        
        return aggregated

class VPE(nn.Module):
    def __init__(self, channels=1536, num_prototypes=32, temperature=0.1):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, channels, 1, 1)
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        x_flat = x.view(b, c, -1)
        
        x_norm = F.normalize(x_flat, p=2, dim=1)
        
        x_norm_fp32 = x_norm.float()
        prototypes_fp32 = self.prototypes.view(self.num_prototypes, c, 1).float()
        
        sim = torch.einsum('bch, pch -> bph', x_norm_fp32, prototypes_fp32)
        
        weights = F.softmax(sim / self.temperature, dim=1)
        
        weights = weights.to(x_flat.dtype)
        
        aggregated = torch.bmm(weights, x_flat.transpose(1, 2))
        
        return aggregated

class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder,num_clusters, vocab_size, max_len, padding_idx, text_d_model, visual_dim=769, clip_model_name="ViT-B/32"):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.text_d_model = text_d_model
        self.num_clusters=num_clusters
        self.padding_idx = padding_idx
        self.visual_dim = visual_dim
        self.word_emb = nn.Embedding(vocab_size, self.visual_dim, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, self.visual_dim, 0), freeze=True)
        self.VPE = VPE(channels=self.visual_dim, num_prototypes=self.num_clusters, temperature=0.1)
        self.TPA = TPA(channels=self.visual_dim, num_prototypes=self.num_clusters, temperature=0.1)
        self.text_proj = nn.Linear(text_d_model if text_d_model is not None else 512, self.visual_dim)
        self.proto_cross_attn = nn.MultiheadAttention(embed_dim=self.visual_dim, num_heads=8, batch_first=True)
        self.text_cross_attn = nn.MultiheadAttention(embed_dim=self.visual_dim, num_heads=8, batch_first=True)
        self.text_encoder = CLIPTextEncoder(model_name=clip_model_name) # Pass model_name
        
        self.itm_head = nn.Linear(self.visual_dim, 2)
        self.tim_head = nn.Linear(self.visual_dim, 2)
        
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, mode, images, seq=None, max_len=None, eos_idx=None, beam_size=None, out_size=1, return_probs=False, text_input=None, epoch=0, compute_decoder=True):
        
        if mode == 'xe':
            if isinstance(images, tuple):
                images = images[0]
            bs, N, C = images.size()
            
            grid_feat = images[:, 1:, :]
            cls_token = images[:, :1, :].squeeze(1)

            x_d = grid_feat.permute(0, 2, 1)
            side_dim = int(math.sqrt(x_d.shape[2]))
            x_d = x_d.reshape(bs, C, side_dim, side_dim)
            similarity_region = self.VPE(x_d).reshape(bs, -1, self.visual_dim)
            
            text_feature_raw = self.text_encoder(text_input)  # (B, text_d_model)
            text_feature = self.text_proj(text_feature_raw)  
            
            pure_text_proto = self.TPA(text_feature.unsqueeze(-1))
            
            
            prob_use_text = 0.5 * max(0, (15 - epoch) / 15.0)
            use_text = torch.rand(1).item() < prob_use_text

            if use_text:
                selected_proto = pure_text_proto 
            else:
                cls_proto = self.TPA(cls_token.unsqueeze(-1))
                selected_proto = cls_proto
            
            similarity_region = similarity_region.float()
            selected_proto = selected_proto.float()
            pure_text_proto = pure_text_proto.float()
            F_v2t_match, _ = self.proto_cross_attn(query=similarity_region, key=pure_text_proto, value=pure_text_proto)
            F_t2v_match, _ = self.text_cross_attn(query=pure_text_proto, key=similarity_region, value=similarity_region)
            
            itm_features = F_v2t_match.mean(dim=1)  # (B, visual_dim)
            itm_logits = self.itm_head(itm_features)   # (B, 2)
            
            tim_features = F_t2v_match.mean(dim=1)  # (B, visual_dim)
            tim_logits = self.tim_head(tim_features)   # (B, 2)

            F_v2t, _ = self.proto_cross_attn(query=similarity_region, key=selected_proto, value=selected_proto)
            F_t2v, _ = self.text_cross_attn(query=selected_proto, key=similarity_region, value=similarity_region)
            fused_features = torch.cat([F_v2t, F_t2v], dim=1)

            dec_output = None
            if compute_decoder:
                grid_enc_output, grid_mask_enc = self.encoder(images)
                pseudo_region_enc_output, pseudo_region_mask_enc = self.encoder(fused_features) # similarity_region
                
                output = torch.cat([grid_enc_output, pseudo_region_enc_output], dim=1)
                mask = torch.cat([grid_mask_enc, pseudo_region_mask_enc], dim=-1)
                dec_output = self.decoder(seq, output, mask)
            
            vis_proto = F.normalize(similarity_region.mean(dim=1), p=2, dim=-1)
            text_proto = F.normalize(pure_text_proto.mean(dim=1), p=2, dim=-1)

            return dec_output, vis_proto, text_proto, cls_token, text_feature, itm_logits, tim_logits

        elif mode == 'rl':
            bs = BeamSearch(self, max_len, eos_idx, beam_size)
            return bs.apply(images, out_size, return_probs)
        
    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]
    
    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                if isinstance(visual, tuple):
                    visual = visual[0]
                
                bs, N, C = visual.size()
                
                grid_feat = visual[:, 1:, :] 
                cls_token = visual[:, :1, :].squeeze(1) # (B, C)

                x_d = grid_feat.permute(0, 2, 1) 
                side_dim = int(math.sqrt(x_d.shape[2]))
                x_d = x_d.reshape(bs, C, side_dim, side_dim)
                similarity_region = self.VPE(x_d).reshape(bs, -1, self.visual_dim) 
                cls_proto = self.TPA(cls_token.unsqueeze(-1))      
                selected_proto = cls_proto 

                similarity_region = similarity_region.float()
                selected_proto = selected_proto.float()

                F_v2t, _ = self.proto_cross_attn(query=similarity_region, key=selected_proto, value=selected_proto)
                F_t2v, _ = self.text_cross_attn(query=selected_proto, key=similarity_region, value=similarity_region)
                fused_features = torch.cat([F_v2t, F_t2v], dim=1)

                grid_enc_output, grid_mask_enc = self.encoder(visual)
                pseudo_region_enc_output, pseudo_region_mask_enc = self.encoder(fused_features)
                self.enc_output = torch.cat([grid_enc_output, pseudo_region_enc_output], dim=1)
                self.mask_enc = torch.cat([grid_mask_enc, pseudo_region_mask_enc], dim=-1)
                
                it = visual.new_full((bs, 1), self.bos_idx).long()
            else:
                it = prev_output
        return self.decoder(it, self.enc_output, self.mask_enc)

class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
