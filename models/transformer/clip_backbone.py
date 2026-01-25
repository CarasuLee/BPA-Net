import torch
import torch.nn as nn
import clip
import os

class CLIPBackbone(nn.Module):
    def __init__(self, model_name="ViT-L/14", device="cpu"):
        super().__init__()
        
        use_transformers = os.path.exists(model_name) or ("/" in model_name and "ViT" not in model_name)
        
        if use_transformers:
            try:
                from transformers import CLIPModel
                print(f"Loading CLIP backbone from Transformers: {model_name}")
                self.model = CLIPModel.from_pretrained(model_name)
                self.model.to(device)
                self.visual = self.model.vision_model
            except Exception as e:
                print(f"Transformers loading failed: {e}. Falling back to OpenAI CLIP.")
                use_transformers = False
        
        if not use_transformers:
            print(f"Loading CLIP backbone from OpenAI CLIP: {model_name}")
            self.model, _ = clip.load(model_name, device=device, jit=False)
            self.model.float()
            self.visual = self.model.visual
            
        self.is_transformers = use_transformers
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, images):
        if self.is_transformers:
            target_dtype = self.visual.embeddings.patch_embedding.weight.dtype
            if images.dtype != target_dtype:
                images = images.to(target_dtype)
            return self.visual(pixel_values=images).last_hidden_state

        target_dtype = self.visual.conv1.weight.dtype
        if images.dtype != target_dtype:
            images = images.type(target_dtype)
            
        # Manually run parts of visual encoder to get sequence output
        x = self.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) # B, dx, w
        
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.visual.ln_post(x)
        return x

    def train(self, mode=True):
        super().train(False)


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="ViT-L/14", device="cpu"):
        super().__init__()
        
        use_transformers = os.path.exists(model_name) or ("/" in model_name and "ViT" not in model_name)

        if use_transformers:
            try:
                from transformers import CLIPModel, CLIPTokenizer
                print(f"Loading CLIP text encoder from Transformers: {model_name}")
                self.model = CLIPModel.from_pretrained(model_name)
                self.model.to(device)
                self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            except Exception as e:
                print(f"Transformers loading failed: {e}. Falling back to OpenAI CLIP.")
                use_transformers = False
        
        if not use_transformers:
            self.model, _ = clip.load(model_name, device=device, jit=False)

        self.is_transformers = use_transformers
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, text):
        device = next(self.parameters()).device
        
        if self.is_transformers:
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
            return self.model.get_text_features(**inputs).float()

        text_tokens = clip.tokenize(text, truncate=True).to(device)
        return self.model.encode_text(text_tokens).float()

    def train(self, mode=True):
        super().train(False)
