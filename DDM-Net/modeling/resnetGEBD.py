import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import copy
import ipdb
from torchvision.models.utils import load_state_dict_from_url
from .resnet import ResNet18
from .transformer import Transformer
from .co_transformer import Co_Transformer
from .position_embedding import PositionEmbeddingSine

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet_Feature_Extractor(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet_Feature_Extractor, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode="bilinear") + y

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x2, x3, x4

    def forward(self, x):
        return self._forward_impl(x)


# resnet18 feature extractor before pool layer
def resnet18_feature_extractor(pretrained=True, progress=True, **kwargs):
    model = ResNet_Feature_Extractor(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet18"], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


# resnet34 feature extractor before pool layer
def resnet34_feature_extractor(pretrained=True, progress=True, **kwargs):
    model = ResNet_Feature_Extractor(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet34"], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


# resnet50 feature extractor before pool layer
def resnet50_feature_extractor(pretrained=True, progress=True, **kwargs):
    model = ResNet_Feature_Extractor(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet50"], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def pairwise_cosine_similarity(x, y):
    x = x.unsqueeze(3)
    y = y.unsqueeze(3)
    x = x.detach()
    y = y.permute(0, 1, 3, 2)
    dot = torch.matmul(x, y)
    x_dist = torch.norm(x, p=2, dim=3, keepdim=True)
    y_dist = torch.norm(y, p=2, dim=2, keepdim=True)
    dist = x_dist * y_dist
    cos = dot / (dist + 1e-8)
    # cos_dist = 1 - cos
    return cos


def pairwise_minus_l2_distance(x, y):
    # x, y: (bs, num_layers, num_frames, num_channels)
    x = x.unsqueeze(3).detach()
    # ([4, 3, 100, 256]) -> ([4, 3, 1, 100, 256])
    y = y.unsqueeze(2)
    l2_dist = torch.sqrt(torch.sum((x - y) ** 2, dim=-1) + 1e-8)
    # (bs, num_layers, num_frames, feature_length)
    l2_dist = nn.InstanceNorm2d(l2_dist.size(1))(l2_dist)
    return -l2_dist


class resnetGEBD(nn.Module):
    def __init__(
        self, backbone="resnet50", pretrained=True, num_classes=2, frames_per_side=5
    ):
        super(resnetGEBD, self).__init__()
        self.num_classes = num_classes
        self.frames_per_side = frames_per_side
        if backbone.lower() == "resnet50":
            self.backbone = resnet50_feature_extractor(pretrained=pretrained)
        elif backbone.lower() == "resnet34":
            self.backbone = resnet34_feature_extractor(pretrained=pretrained)
        elif backbone.lower() == "resnet18":
            self.backbone = resnet18_feature_extractor(pretrained=pretrained)
        else:
            raise NotImplementedError("{} not supported!".format(backbone))

        del self.backbone.avgpool, self.backbone.fc

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(in_features=512, out_features=self.num_classes, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=self.num_classes, bias=True)

        # ddm
        self.encoder_dim = 512
        self.hidden_dim_1d = 512
        self.ddm_encoder = ResNet18(9, self.hidden_dim_1d)

        self.short_conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.encoder_dim, self.encoder_dim, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(self.encoder_dim, self.encoder_dim, 1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(3)
            ]
        )

        self.middle_conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.encoder_dim, self.encoder_dim, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(self.encoder_dim, self.encoder_dim, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(3)
            ]
        )

        self.long_conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.encoder_dim, self.encoder_dim, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(self.encoder_dim, self.encoder_dim, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(3)
            ]
        )

        self.ddm_decoder = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, 1),
        )

        self.proj_rgb = nn.Sequential(
            nn.Linear(512, 512, bias=True), nn.ReLU(inplace=True)
        )
        self.proj_ddm = nn.Sequential(
            nn.Linear(512, 512, bias=True), nn.ReLU(inplace=True)
        )
        self.proj_v = nn.Sequential(nn.Linear(512, 1, bias=True), nn.ReLU(inplace=True))

        self.x3_out = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
        )
        self.x4_out = nn.Sequential(
            nn.Conv1d(2048, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
        )

        self.alpha = nn.parameter.Parameter(
            torch.zeros(
                6,
            )
        )

        self.intra_transformer1 = Transformer()
        self.intra_transformer2 = Transformer()
        self.cross_transformer1 = Co_Transformer()
        self.cross_transformer2 = Co_Transformer()

        self.pos = PositionEmbeddingSine()
        self.pos2 = PositionEmbeddingSine(num_locations=5)

    def forward(self, x):
        # (bs, num_frames, C, H, W)
        B = x.shape[0]
        T = x.shape[1]
        x = einops.rearrange(x, "b t c h w -> (b t) c h w")

        # layer1 (bs*num_frames, 256, 56, 56)
        # layer2 (bs*num_frames, 512, 28, 28)
        # layer3 (bs*num_frames, 1024, 14, 14)
        # layer4 (bs*num_frames, 2048, 7, 7)
        x2, x3, x4 = self.backbone(x)

        x2 = self.avg_pool(x2).squeeze()
        x3 = self.avg_pool(x3).squeeze()
        x4 = self.avg_pool(x4).squeeze()

        x2 = einops.rearrange(x2, "(b f) c -> b f c", b=B).permute(0, 2, 1)
        x3 = einops.rearrange(x3, "(b f) c -> b f c", b=B).permute(0, 2, 1)
        x3 = self.x3_out(x3)
        x4 = einops.rearrange(x4, "(b f) c -> b f c", b=B).permute(0, 2, 1)
        x4 = self.x4_out(x4)

        short_feat = 0.5 * (self.short_conv[0](x2) + x2)
        middle_feat = 0.5 * (self.middle_conv[0](x2) + x2)
        long_feat = 0.5 * (self.long_conv[0](middle_feat) + middle_feat)
        out_list = [short_feat, middle_feat, long_feat]

        short_feat = 0.5 * (self.short_conv[1](x3) + x3)
        middle_feat = 0.5 * (self.middle_conv[1](x3) + x3)
        long_feat = 0.5 * (self.long_conv[1](middle_feat) + middle_feat)
        out_list.extend([short_feat, middle_feat, long_feat])

        short_feat = 0.5 * (self.short_conv[2](x4) + x4)
        middle_feat = 0.5 * (self.middle_conv[2](x4) + x4)
        long_feat = 0.5 * (self.long_conv[2](middle_feat) + middle_feat)
        out_list.extend([short_feat, middle_feat, long_feat])
        # (bs, num_layers, num_frames, C)
        out = torch.stack(out_list, dim=1).permute(0, 1, 3, 2)

        # (bs, num_layers, num_frames, num_frames)
        ddm = pairwise_minus_l2_distance(out, out)

        # (bs, C, num_frames, num_frames)
        ddm = self.ddm_encoder(ddm)

        ddm_feat = []
        for t1 in range(T):
            ddm = ddm.permute(0, 2, 3, 1)
            # proj_rgb: (bs, 128)
            # proj_ddm: (bs, num_frames, 128)
            # (bs, num_frames, 1) -> (bs, num_frames)
            attn_weight = self.proj_v(
                torch.tanh(
                    self.proj_rgb(x4[:, :, t1]).unsqueeze(dim=1)
                    + self.proj_ddm(ddm[:, :, t1, :])
                )
            ).squeeze()
            attn_weight = F.softmax(attn_weight, dim=1)

            ddm = ddm.permute(0, 3, 1, 2)
            # (bs, C, num_frames) * (bs, num_frames, 1)
            ddm_feature_t = torch.bmm(
                ddm[:, :, t1, :], attn_weight.unsqueeze(dim=2)
            ).squeeze()
            ddm_feat.append(ddm_feature_t.unsqueeze(dim=2))
        # (bs, C, num_frames)
        ddm_feat = torch.cat(ddm_feat, dim=2)
        # (bs, C, num_frames)
        ddm_feat = self.ddm_decoder(ddm_feat)

        # (bs, num_frames, 512)
        pos = self.pos(ddm_feat.permute(0, 2, 1))
        intra_rgb_feat = self.intra_transformer1(x4, pos)[-1].permute(0, 2, 1)
        intra_ddm_feat = self.intra_transformer2(ddm_feat, pos)[-1].permute(0, 2, 1)

        cross_rgb_feat = self.cross_transformer1(intra_rgb_feat, intra_ddm_feat)
        cross_ddm_feat = self.cross_transformer2(intra_ddm_feat, intra_rgb_feat)

        results = []
        rgbs = []
        ddms = []
        for layer_idx in range(cross_rgb_feat.shape[0]):
            rgb_feat_layer = cross_rgb_feat[layer_idx]
            ddm_feat_layer = cross_ddm_feat[layer_idx]

            rgb_feat_layer = torch.mean(rgb_feat_layer, dim=1)
            rgb_feat_layer = rgb_feat_layer.flatten(1)
            rgb_feat_layer = self.fc(rgb_feat_layer)

            ddm_feat_layer = torch.mean(ddm_feat_layer, dim=1)
            ddm_feat_layer = ddm_feat_layer.flatten(1)
            ddm_feat_layer = self.fc2(ddm_feat_layer)

            alpha = torch.sigmoid(self.alpha[layer_idx]).unsqueeze(0)
            result = (1 - alpha) * rgb_feat_layer + alpha * ddm_feat_layer
            results.append(result)
            rgbs.append(rgb_feat_layer)
            ddms.append(ddm_feat_layer)

        return results, rgbs, ddms
