import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    avg_pool_nd
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


class ControlledLiteUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        # with torch.no_grad():
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        input_layers_indx_to_add = [1,4,7,10]
        for i, module in enumerate(self.input_blocks):
            if control is not None and i in input_layers_indx_to_add:
                h += control.pop(0)
            h = module(h, emb, context)
            hs.append(h)

        if control is not None:
            h += control.pop(0)

        h = self.middle_block(h, emb, context)

        for i, module in enumerate(self.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlledLiteDecoderUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        input_layers_indx_to_add = [0,5,8,9]
        for i, module in enumerate(self.output_blocks):
            if not  only_mid_control and control is not None and i in input_layers_indx_to_add:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlLiteNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels, # 3
            num_res_blocks, # 2
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            control_type='mlp',
            use_diffusion_input=False
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims # 2, for 2D convolutions
        self.image_size = image_size # 32
        self.in_channels = in_channels # 4
        self.model_channels = model_channels # 320
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks # [2,2,2,2]
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions # [4,2,1]
        self.dropout = dropout
        self.channel_mult = channel_mult # [1,2,4,4]
        self.conv_resample = conv_resample # True
        self.use_checkpoint = use_checkpoint # True
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads # 8
        self.num_head_channels = num_head_channels # -1
        self.num_heads_upsample = num_heads_upsample # 8
        self.predict_codebook_ids = n_embed is not None
        self.control_type = control_type
        self.use_diffusion_input = use_diffusion_input

        time_embed_dim = model_channels * 4  # 1280
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            # zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
           conv_nd(dims, 256, model_channels, 3, padding=1)

        )

        self._feature_size = self.model_channels

        if use_diffusion_input:
            self.input_blocks_control = nn.ModuleList(
                [
                    TimestepEmbedSequential(
                        conv_nd(dims, in_channels, model_channels, 3, padding=1)
                    )
                ]
            )
            self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        else:
            self.input_blocks_control = nn.ModuleList([])
            self.zero_convs = nn.ModuleList([])

        if self.control_type == 'mlp':
            self.init_mlp_2()
        elif self.control_type == 'conv':
            self.init_conv()

        # # Middle block
        # mlp = TimestepEmbedSequential(
        #     conv_nd(dims, ch, ch, 1, padding=0),
        #     nn.SiLU(),
        # )
        #
        # self.input_blocks_control.append(mlp)
        # self.zero_convs.append(self.make_zero_conv(ch))
        # self._feature_size += ch
        # input_block_chans.append(ch)

    def init_mlp_2(self):
        ch = self.model_channels
        strides = [2, 2, 2, 1]
        for level, (mult, stride,) in enumerate(zip(self.channel_mult, strides)):
            mlp = TimestepEmbedSequential(
                conv_nd(self.dims, ch, self.model_channels * mult, 1, padding=0),
                nn.SiLU(),
                avg_pool_nd(self.dims, kernel_size=stride, stride=stride)
                # stride=2 is used, even though diagram specific 3x downsample

            )

            self.input_blocks_control.append(mlp)

            ch = mult * self.model_channels
            self.zero_convs.append(self.make_zero_conv(ch))
            self._feature_size += ch

    def init_mlp(self):
        ch = self.model_channels
        size = self.image_size
        strides = [2, 2, 2, 1]
        for level, (mult, stride,) in enumerate(zip(self.channel_mult, strides)):
            mlp = TimestepEmbedSequential(
                nn.Flatten(start_dim=1),
                linear(size ** 2 * ch, size ** 2 * self.model_channels * mult),
                nn.SiLU(),
                avg_pool_nd(self.dims, kernel_size=stride, stride=stride),
                nn.Unflatten(1, (self.model_channels * mult, size // stride, size // stride))

            )

            self.input_blocks_control.append(mlp)

            ch = mult * self.model_channels
            size = size // stride
            self.zero_convs.append(self.make_zero_conv(ch))
            self._feature_size += ch

    def init_conv(self):
        ch = self.model_channels
        strides = [2, 2, 2, 1]
        for level, (mult, stride,) in enumerate(zip(self.channel_mult, strides)):
            mlp = TimestepEmbedSequential(
                conv_nd(self.dims, ch, self.model_channels * mult, 3, padding=1),
                nn.SiLU(),
                avg_pool_nd(self.dims, kernel_size=stride, stride=stride)
                # stride=2 is used, even though diagram specific 3x downsample

            )

            self.input_blocks_control.append(mlp)

            ch = mult * self.model_channels
            self.zero_convs.append(self.make_zero_conv(ch))
            self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb) # EMBDS ARE NOT USED, JUST FOR COMPATIBILITY!!!!!
        #[bs, 1280]

        guided_hint = self.input_hint_block(hint, emb, context)

        if self.use_diffusion_input:
            h = x.type(self.dtype)
            outs = []
        else:
            outs = [guided_hint]
            h = guided_hint
            guided_hint = None

        for i, (module, zero_conv) in enumerate(zip(self.input_blocks_control, self.zero_convs)):
            h = module(h, emb, context)
            if guided_hint is not None:
                h = h + guided_hint
                guided_hint = None
            outs.append(zero_conv(h, emb, context))

        return outs