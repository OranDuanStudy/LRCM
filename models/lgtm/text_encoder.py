import clip
from clip.model import CLIP
from torch import Tensor, nn


def available_models():
    return clip.available_models()


def load_and_freeze_clip(clip_model_name: str, device="cpu") -> CLIP:
    clip_model: CLIP = clip.load(clip_model_name ,device=device, jit=False)[0]
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    return clip_model




class CLIP_TextEncoder(nn.Module):
    def __init__(self, clip_model_name="ViT-B/32", output_dim=512, device="cpu"):
        super().__init__()
        self.model = load_and_freeze_clip(clip_model_name, device=device)
        
        #self.output_dim = self.model.transformer.width
        self.output_dim = output_dim
        self.device = device

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def encode(self, texts: list[str]):
        device = self.device # next(self.parameters()).device
        tokens = clip.tokenize(texts, truncate=True).to(device)
        x: Tensor = self.model.encode_text(tokens)
        x = x.to(device)  # Ensure tensor is on the correct device
        return x.detach()  # Detach tensor

    def forward(self, texts: list[str]):
        return self.encode(texts)


if __name__ == "__main__":
    # Test with plain text
    text = "This is a test sentence for CLIP text encoder."

    clip_models = CLIP_TextEncoder("ViT-B/32")
    print(clip_models.encode([text]))
    