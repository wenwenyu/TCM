from collections import OrderedDict
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path
from functools import partial

from mmrotate.registry import MODELS


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, adapter=False):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

        self.adapter = adapter
        if adapter:
            self.adapter = ConvResidualAdapter(inplanes)

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.adapter:
            _adapter = self.adapter(x)
            out += _adapter

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        # print(f'the shape of attention pool :{x.shape}')  # ([1, 5, 640]), ([197, 5, 640])
        x = x.permute(1, 2, 0)
        global_feat = x[:, :, 0] # ([5, 640])
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)  # ([5, 640, 14, 14])
        return global_feat, feature_map


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, use_clip_ckpt=True, adapter=False, side_adapter_width=[]):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.use_clip_ckpt = use_clip_ckpt
        if use_clip_ckpt:
            # the 3-layer stem
            self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(width // 2)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(width // 2)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(width)
            self.relu3 = nn.ReLU(inplace=True)
            self.avgpool = nn.AvgPool2d(2)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.adapter = adapter
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

        # adapter
        # x[:4]将每个stage的输出进行adapter 调整和resnet的输出维度一致
        # x[4] r50 (B, 1024, 32, 32) to samvitb (B, 256, 64, 64)
        self.side_adapter = True if side_adapter_width else False
        if self.side_adapter:
            self.side_bridges = nn.ModuleList()
            for i in range(len(side_adapter_width)):
                in_channels = side_adapter_width[i]
                out_channels = width * 2 ** i * Bottleneck.expansion
                side_bridge = ChannelAdapter(in_channels, out_channels)
                self.side_bridges.append(side_bridge)
            
            attnpool_bridge = nn.Sequential(
                nn.Conv2d(output_dim, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                nn.BatchNorm2d(256),
                nn.GELU(),
            )
            self.side_bridges.append(attnpool_bridge)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride, adapter=self.adapter)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, adapter=self.adapter))

        return nn.Sequential(*layers)

    def forward(self, x, side_x=None):
        if self.use_clip_ckpt:
            def stem(x):
                x = self.relu1(self.bn1(self.conv1(x)))
                x = self.relu2(self.bn2(self.conv2(x)))
                x = self.relu3(self.bn3(self.conv3(x)))
                x = self.avgpool(x)
                return x
        else:
            def stem(x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.maxpool(x)
                return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)

        outs = []
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        for i, layer in enumerate(layers):
            x = layer(x)
            if self.side_adapter:
                x = x + self.side_bridges[i](side_x[i])
            outs.append(x)

        x_global, x_local = self.attnpool(x)
        if self.side_adapter:
            x_local = self.side_bridges[4](x_local)
            x_local = x_local + side_x[4]
        outs.append([x_global, x_local])

        return tuple(outs)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, adapter: bool = False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.adapter = adapter
        if adapter:
            self.adapter_1 = ViTResidualAdapter(d_model)
            self.adapter_2 = ViTResidualAdapter(d_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        att =  self.attention(self.ln_1(x))
        if self.adapter:
            att = self.adapter_1(att)
        x = x + att
        mlp = self.mlp(self.ln_2(x))
        if self.adapter:
            mlp = self.adapter_2(mlp)
        x = x + mlp
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
    

class Transformer_fpn(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, adapter: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, adapter=adapter) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        outputs = []
        for resblock in self.resblocks:
            x = resblock(x)
            outputs.append(x)
        return tuple(outputs)
    


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


class FPNInputAdapter(nn.Module):
    def __init__(self, input_resolution: int,  vision_patch_size, vision_layers, vision_width, embed_dim, output_levels: int = 4):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = vision_patch_size
        self.layers = vision_layers
        self.width = vision_width
        self.output_levels = output_levels
        self.output_wh = [input_resolution//2**(i+1) for i in range(1, output_levels+1)]  # [256, 128, 64, 32]

        self.UpConv_4x = nn.Sequential(
            nn.GroupNorm(1, vision_width),
            nn.ConvTranspose2d(vision_width, vision_width, kernel_size=2, stride=2),
            nn.BatchNorm2d(vision_width),
            nn.GELU(),
            nn.ConvTranspose2d(vision_width, vision_width, kernel_size=2, stride=2),
        )
        self.UpConv_2x = nn.Sequential(
            nn.GroupNorm(1, vision_width),
            nn.ConvTranspose2d(vision_width, vision_width, kernel_size=2, stride=2),
        )
        self.UpConv_1x = nn.GroupNorm(1, vision_width)
        self.DwConv_2x = nn.Sequential(
            nn.GroupNorm(1, vision_width),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.apply(weights_init)


    def forward(self, vt_outputs):
        selected_outputs = []
        for i in range(self.output_levels):
            idx = int(self.layers * (i+1) / (self.output_levels))-1  # [3, 6, 9, 12]
            selected_outputs.append(vt_outputs[idx])

        adapted_outputs = []
        patch_size = self.patch_size
        hi, wi = self.input_resolution // patch_size, self.input_resolution // patch_size
        for out, ow in zip(selected_outputs, self.output_wh):
            out = out[:, 1:, :]  # Remove class token
            out = out.permute(0, 2, 1)  # NLD -> NDL
            out = out.reshape(out.shape[0], out.shape[1], hi, wi)
            
            if patch_size == 14:
                out = F.interpolate(out, scale_factor=patch_size//16, mode='bilinear')
                
            if ow==256:
                out = self.UpConv_4x(out)
            elif ow==128:
                out = self.UpConv_2x(out)
            elif ow==64:
                out = self.UpConv_1x(out)
            elif ow==32:
                out = self.DwConv_2x(out)
            else:
                raise ValueError("Unsupported output width for upsampling.")
            if out.shape[-1] != ow:
                out = F.interpolate(out, size=(ow, ow), mode='bilinear')
            
            adapted_outputs.append(out)
        
        atten_pool_out = vt_outputs[-1]
        global_feat = atten_pool_out[0]  # ([2, 512])
        feature_map = atten_pool_out[1].permute(0, 2, 1).reshape(global_feat.shape[0], global_feat.shape[1], hi, wi)  # ([2, 512, 64, 64])
        if patch_size==14:
            feature_map = F.interpolate(feature_map, (64, 64), mode='bilinear')

        adapted_outputs.append([global_feat, feature_map])

        return tuple(adapted_outputs)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, adapter: bool):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer_fpn(width, layers, heads, adapter=adapter)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = [xi.permute(1, 0, 2) for xi in x]  # LND -> NLD, for each output in the tuple  [2, 4097, 768]

        features = [self.ln_post(xi) for xi in x]  # Apply ln_post to each output in the tuple
        x_last = features[-1] @ self.proj
        global_embedding = x_last[:, 0]
        visual_embedding = x_last[:, 1:]
        features.append([global_embedding, visual_embedding])

        return tuple(features)

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


@MODELS.register_module()
class VisualEncoder(nn.Module):
    def __init__(self, pretrained_clip_path: Optional[Union[str, Path]]=None, image_resolution: int=1024,
                 freeze=False, fp16=False, load_weight=True, adapter=False, use_clip_ckpt=True, side_adapter_width=[], same_ABtype=False, **kwargs):
        super().__init__()

        # Load the state_dict from the pre-trained CLIP model
        if pretrained_clip_path is not None:
            if use_clip_ckpt:
                state_dict = torch.jit.load(pretrained_clip_path, map_location='cpu').state_dict()
                # Check if the model is based on VisionTransformer or ModifiedResNet
                self.vit = "visual.proj" in state_dict
                # Extract relevant parameters
                embed_dim = state_dict["text_projection"].shape[1]
            else:
                state_dict = torch.load(pretrained_clip_path, map_location=torch.device('cpu'))
                if 'resnet50' in pretrained_clip_path:  # torchvision resnet50
                    self.vit = False
                else:  # sam vit
                    self.vit = True
        else:
            self.vit = True
            use_clip_ckpt = False
            state_dict = None

        self.use_clip_ckpt = use_clip_ckpt
        
        if self.vit:
            if use_clip_ckpt:
                vision_width = state_dict["visual.conv1.weight"].shape[0]
                vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
                vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
                self.vision_patch_size = vision_patch_size
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_width // 64,
                    output_dim=embed_dim,
                    adapter=adapter
                )
                self.fpn_input_adapter = FPNInputAdapter(image_resolution, vision_patch_size, vision_layers, vision_width, embed_dim)

        else:
            if use_clip_ckpt:
                counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
                vision_layers = tuple(counts)
                vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
                self.visual = ModifiedResNet(
                    layers=vision_layers,
                    output_dim=embed_dim,
                    heads=vision_width * 32 // 64,
                    input_resolution=image_resolution,
                    width=vision_width,
                    adapter=adapter,
                    side_adapter_width=side_adapter_width
                )
            else:
                self.visual = ModifiedResNet(
                    layers=(3, 4, 6, 3),
                    output_dim=1024,
                    heads=32,
                    input_resolution=image_resolution,
                    width=64,
                    use_clip_ckpt=False,
                    adapter=adapter,
                    side_adapter_width=side_adapter_width
                )

        # Load the pre-trained weights
        self.freeze = freeze
        self.fp16 = fp16
        if load_weight and state_dict is not None:
            self._init_weights(state_dict)
        else:
            self.dtype = torch.float16 if fp16 else torch.float32

    def _init_weights(self, state_dict):
        # Filter the state_dict to only include visual model weights
        if self.use_clip_ckpt:
            visual_weights = {k[len("visual."):]: v for k, v in state_dict.items() if k.startswith("visual.")}
        else:
            if self.vit:
                visual_weights = {k[len("image_encoder."):]: v for k, v in state_dict.items() if k.startswith("image_encoder.")}
            else:
                visual_weights = state_dict
        if isinstance(self.visual, VisionTransformer):
            if 'positional_embedding' in visual_weights.keys() and\
                visual_weights['positional_embedding'].shape != self.visual.positional_embedding.shape:
                print(f'Resize the VisionTransformer pos_embed shape from {visual_weights["positional_embedding"].shape} to {self.visual.positional_embedding.shape}')
                # Calculate the new number of patches
                num_patches = int(self.visual.input_resolution // self.vision_patch_size)
                # Resize positional_embedding
                old_pos_embed = visual_weights['positional_embedding']
                # Get the original number of patches
                old_num_patches = int((old_pos_embed.shape[0] - 1) ** 0.5)
                # Reshape and transpose old_pos_embed[1:]
                old_pos_embed_reshaped = old_pos_embed[1:].view(old_num_patches, old_num_patches, -1).permute(2, 0, 1)
                new_pos_embed = F.interpolate(
                    old_pos_embed_reshaped.unsqueeze(0),
                    size=(num_patches, num_patches),
                    mode='bicubic'
                ).squeeze(0)
                # Reshape and transpose back to the original format
                new_pos_embed = new_pos_embed.permute(1, 2, 0).view(-1, old_pos_embed.shape[-1])
                visual_weights['positional_embedding'] = torch.cat([old_pos_embed[0].unsqueeze(0), new_pos_embed], dim=0)

        elif isinstance(self.visual, ModifiedResNet):
            if 'attnpool.positional_embedding' in visual_weights.keys() and\
                visual_weights['attnpool.positional_embedding'].shape != self.visual.attnpool.positional_embedding.shape:
                print(f'Resize the ModifiedResNet pos_embed shape from {visual_weights["attnpool.positional_embedding"].shape} to {self.visual.attnpool.positional_embedding.shape}')
                old_pos_embed = visual_weights['attnpool.positional_embedding']
                # Calculate the new spatial_dim
                spatial_dim = self.visual.input_resolution // 32
                # Get the original spatial_dim
                old_spatial_dim = int((old_pos_embed.shape[0] - 1) ** 0.5)
                # Reshape and transpose old_pos_embed[1:]
                old_pos_embed_reshaped = old_pos_embed[1:].view(old_spatial_dim, old_spatial_dim, -1).permute(2, 0, 1)
                # Resize positional_embedding
                new_pos_embed = F.interpolate(
                    old_pos_embed_reshaped.unsqueeze(0),
                    size=(spatial_dim, spatial_dim),
                    mode='bicubic'
                ).squeeze(0)
                # Reshape and transpose back to the original format
                new_pos_embed = new_pos_embed.permute(1, 2, 0).view(-1, old_pos_embed.shape[-1])
                visual_weights['attnpool.positional_embedding'] = torch.cat([old_pos_embed[0].unsqueeze(0), new_pos_embed], dim=0)
                
        # If necessary, convert weights to fp16
        if self.fp16:
            convert_weights(self.visual)
            self.dtype = torch.float16
        else:
            visual_weights = {k: v.float() for k, v in visual_weights.items()}
            self.dtype = torch.float32
        
        # Load the visual model weights
        print("Loading visual model weights...")
        missing, unexpected = self.visual.load_state_dict(visual_weights, strict=False)
        print(missing, unexpected, 'are misaligned params in visual encoder')
        # 冻结已加载权重的网络部分
        if self.freeze:
            for name, param in self.visual.named_parameters():
                if name in visual_weights:
                    param.requires_grad = False

    def forward(self, image, side_x=None):
        if isinstance(self.visual, VisionTransformer):
            vt_outputs = []
            for vt_out in self.visual(image.type(self.dtype)):
                if isinstance(vt_out, torch.Tensor):
                    vt_outputs.append(vt_out.type(torch.float32))
                elif isinstance(vt_out, list):
                    vt_outputs.append([j.type(torch.float32) for j in vt_out])
            fpn_inputs = self.fpn_input_adapter(vt_outputs)
            return fpn_inputs
        
        elif isinstance(self.visual, ModifiedResNet):
            outs = []
            for out in self.visual(image.type(self.dtype), side_x):
                if isinstance(out, torch.Tensor):
                    outs.append(out.type(torch.float32))
                elif isinstance(out, list):
                    outs.append([j.type(torch.float32) for j in out])
            return tuple(outs)

        else:
            raise NotImplementedError


@MODELS.register_module()
class TextEncoder(nn.Module):
    def __init__(self, pretrained_clip_path: Union[str, Path], context_length=None, freeze=True, **kwargs):
        super().__init__()
        ''' Text encoder based on the CLIP model
            context_length: int,
            vocab_size: int,
            transformer_width: int,
            transformer_heads: int,
            transformer_layers: int
        '''

        # Load the state_dict from the pre-trained CLIP model
        state_dict = torch.jit.load(pretrained_clip_path, map_location='cpu').state_dict()
        self.dtype = state_dict['visual.conv1.weight'].dtype

        # Extract relevant parameters
        embed_dim = state_dict["text_projection"].shape[1]
        self.embed_dim = embed_dim
        if context_length is None:
            context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        self.transformer_width = transformer_width
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

        # Build the text transformer
        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        # Build other required layers and parameters
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # Initialize the parameters
        self.freeze = freeze
        self._init_weights(state_dict)

    def _init_weights(self, state_dict):
        self.token_embedding.weight.data = state_dict["token_embedding.weight"].clone().float()
        self.ln_final.weight.data = state_dict["ln_final.weight"].clone().float()
        self.ln_final.bias.data = state_dict["ln_final.bias"].clone().float()
        self.text_projection.data = state_dict["text_projection"].clone().float()

        if self.context_length < state_dict["positional_embedding"].shape[0]:
            print('positional_embedding is tuncated from 77 to', self.context_length)
            self.positional_embedding.data = state_dict["positional_embedding"][:self.context_length].clone().float()
        else:
            self.positional_embedding.data = state_dict["positional_embedding"].clone().float()
        transformer_weights = {k[len("transformer."):]: v.float() for k, v in state_dict.items() if k.startswith("transformer.")}
        
        # Load the visual model weights
        print("Loading text transformer weights...")
        missing, unexpected = self.transformer.load_state_dict(transformer_weights, strict=False)
        print(missing, unexpected, 'are misaligned params in text encoder')

        # 冻结已加载权重的网络部分
        if self.freeze:
            for name, param in self.transformer.named_parameters():
                if name in transformer_weights:
                    param.requires_grad = False
            self.token_embedding.weight.requires_grad = False
            self.ln_final.weight.requires_grad = False
            self.ln_final.bias.requires_grad = False
            self.text_projection.requires_grad = False
            self.positional_embedding.requires_grad = False
    
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, context, prompt_gen=None):
        x_text = self.token_embedding(text)   # [batch_size, n_ctx, d_model]
        
        K, N1, C = x_text.shape
        B, N2, C = context.shape

        x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
        context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)

        x = torch.cat([x_text[:,:,0:1], context, x_text[:, :, 1:]], dim=2).reshape(B*K, N1+N2, C)

        if prompt_gen is not None:
            B = prompt_gen.shape[0]
            prompt_gen = prompt_gen.reshape(B, 1, C).expand(B, K, C).reshape(B*K, C)
            x = x.unsqueeze(0).expand(B, -1, -1, -1).reshape(B*K, N1+N2, C) + prompt_gen.reshape(B*K, 1, C)

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eos_indx = text.argmax(dim=-1) + N2
        eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)
        x = x[torch.arange(x.size(0)), eos_indx] # NLD -> ND
        # project from transformer width to embedding dimension
        x = torch.matmul(x, self.text_projection.float()) # [batch_size, d_model]
        x = x.reshape(B, K, self.embed_dim)
        return x
    

@MODELS.register_module()
class ContextDecoder(nn.Module):
    def __init__(self,
                 pretrained_clip_path: Union[str, Path]=None,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        if pretrained_clip_path is None:
            visual_dim=256
        else:
            state_dict = torch.jit.load(pretrained_clip_path, map_location='cpu').state_dict()
            visual_dim = state_dict["text_projection"].shape[1]

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width))

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width))

        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual)
        
        return self.out_proj(x)

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, mem, mem)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class UpsampleBlock_PixelShuffle(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock_PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels * (scale_factor ** 2))
        self.activate = QuickGELU()
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        x = self.pixel_shuffle(x)
        return x
    

class UpsampleBlock_TransConv(nn.Module):
    def __init__(self, in_channels):
        super(UpsampleBlock_TransConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(in_channels)
        self.activate = QuickGELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = QuickGELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x
    

class ConvResidualAdapter(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        hidden_dim = int(in_channels // 4)
        self.down_project = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            QuickGELU(), 
        )
        self.up_project = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            QuickGELU(), 
        )

    def forward(self, x: torch.Tensor):
        identity = x
        x = self.down_project(x)
        x = self.up_project(x)
        x += identity
        return x


class ViTResidualAdapter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        hidden_dim = int(d_model // 4)
        self.down_project = nn.Sequential(
            LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            QuickGELU(), 
        )
        self.up_project = nn.Sequential(
            LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
            QuickGELU(),
        )

    def forward(self, x: torch.Tensor):
        identity = x
        x = self.down_project(x)
        x = self.up_project(x)
        x += identity
        return x


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAdapter, self).__init__()

        self.adapter = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        return self.adapter(x)