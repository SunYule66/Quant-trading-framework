#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: Optional[str]) -> Optional[nn.Module]:
    if not name:
        return None
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "softplus":
        return nn.Softplus()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "linear":
        return None
    raise ValueError("Activation {} is not supported.".format(name))


class LayerOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, prev_w):  # pragma: no cover - subclasses override
        raise NotImplementedError

    def regularization_loss(self):
        return None


class ConvLayerOp(LayerOp):
    def __init__(self, in_channels, layer_cfg):
        super().__init__()
        kernel = tuple(allint(layer_cfg.get("filter_shape", [1, 1])))
        strides = tuple(allint(layer_cfg.get("strides", [1, 1])))
        padding_type = layer_cfg.get("padding", "valid").lower()
        if padding_type == "same":
            padding = (kernel[0] // 2, kernel[1] // 2)
        else:
            padding = (0, 0)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=int(layer_cfg["filter_number"]),
                              kernel_size=kernel,
                              stride=strides,
                              padding=padding)
        self.activation = _activation(layer_cfg.get("activation_function"))
        self.weight_decay = float(layer_cfg.get("weight_decay", 0.0) or 0.0)

    def forward(self, x, prev_w):
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        return out

    def regularization_loss(self):
        if self.weight_decay:
            return self.weight_decay * torch.sum(self.conv.weight.pow(2))
        return self.conv.weight.new_tensor(0.0)


class DropoutOp(LayerOp):
    def __init__(self, keep_probability):
        super().__init__()
        probability = 1 - float(keep_probability)
        self.dropout = nn.Dropout(p=probability)

    def forward(self, x, prev_w):
        return self.dropout(x)


class PoolingOp(LayerOp):
    def __init__(self, pool_type, layer_cfg):
        super().__init__()
        kernel = tuple(allint(layer_cfg.get("strides", [2, 2])))
        if pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size=kernel, stride=kernel)
        else:
            self.pool = nn.AvgPool2d(kernel_size=kernel, stride=kernel)

    def forward(self, x, prev_w):
        return self.pool(x)


class LRNOp(LayerOp):
    def __init__(self):
        super().__init__()
        # values aligned with tflearn defaults
        self.norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1.0)

    def forward(self, x, prev_w):
        return self.norm(x)


class DenseLayerOp(LayerOp):
    def __init__(self, in_features, layer_cfg):
        super().__init__()
        self.linear = nn.Linear(in_features, int(layer_cfg["neuron_number"]))
        self.activation = _activation(layer_cfg.get("activation_function"))
        self.weight_decay = float(layer_cfg.get("weight_decay", 0.0) or 0.0)

    def forward(self, x, prev_w):
        batch = x.shape[0]
        out = x.view(batch, -1)
        out = self.linear(out)
        if self.activation:
            out = self.activation(out)
        # reshape back to N,C,1,1 for downstream conv compatibility
        return out.view(batch, out.shape[1], 1, 1)

    def regularization_loss(self):
        if self.weight_decay:
            return self.weight_decay * torch.sum(self.linear.weight.pow(2))
        return self.linear.weight.new_tensor(0.0)


class EIIEDenseOp(LayerOp):
    def __init__(self, in_channels, current_width, layer_cfg):
        super().__init__()
        kernel_size = (1, current_width)
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=int(layer_cfg["filter_number"]),
                              kernel_size=kernel_size,
                              stride=(1, 1),
                              padding=0)
        self.activation = _activation(layer_cfg.get("activation_function"))
        self.weight_decay = float(layer_cfg.get("weight_decay", 0.0) or 0.0)

    def forward(self, x, prev_w):
        out = self.conv(x)
        if self.activation:
            out = self.activation(out)
        return out

    def regularization_loss(self):
        if self.weight_decay:
            return self.weight_decay * torch.sum(self.conv.weight.pow(2))
        return self.conv.weight.new_tensor(0.0)


class EIIEOutputOp(LayerOp):
    def __init__(self, in_channels, rows, width, layer_cfg):
        super().__init__()
        self.rows = rows
        self.width = width
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels * width, 1)
        self.weight_decay = float(layer_cfg.get("weight_decay", 0.0) or 0.0)

    def forward(self, x, prev_w):
        batch = x.shape[0]
        features = x.permute(0, 2, 1, 3).reshape(batch * self.rows, self.in_channels * self.width)
        logits = self.linear(features).view(batch, self.rows)
        btc_bias = torch.ones((batch, 1), device=x.device)
        weights = torch.cat([btc_bias, logits], dim=1)
        return F.softmax(weights, dim=1)

    def regularization_loss(self):
        if self.weight_decay:
            return self.weight_decay * torch.sum(self.linear.weight.pow(2))
        return self.linear.weight.new_tensor(0.0)


class EIIEOutputWithWOp(LayerOp):
    def __init__(self, in_channels, rows, width, layer_cfg):
        super().__init__()
        self.rows = rows
        self.width = width
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels * width + 1, 1)
        self.weight_decay = float(layer_cfg.get("weight_decay", 0.0) or 0.0)
        # 改进初始化：使用 Xavier 初始化，使初始权重更有利于学习
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        nn.init.zeros_(self.linear.bias)
        # btc_bias 初始化为小的随机值，而不是 0
        self.register_parameter("btc_bias", nn.Parameter(torch.randn(1) * 0.1))

    def forward(self, x, prev_w):
        batch = x.shape[0]
        features = x.permute(0, 2, 1, 3).reshape(batch * self.rows, self.in_channels * self.width)
        prev = prev_w.reshape(batch * self.rows, 1)
        stacked = torch.cat([features, prev], dim=1)
        logits = self.linear(stacked).view(batch, self.rows)
        btc_bias = self.btc_bias.expand(batch, 1)
        weights = torch.cat([btc_bias, logits], dim=1)
        return F.softmax(weights, dim=1)

    def regularization_loss(self):
        if self.weight_decay:
            return self.weight_decay * torch.sum(self.linear.weight.pow(2))
        return self.linear.weight.new_tensor(0.0)


class OutputWithWOp(LayerOp):
    def __init__(self, in_features, rows, layer_cfg):
        super().__init__()
        self.rows = rows
        self.linear = nn.Linear(in_features + rows, rows + 1)
        self.weight_decay = float(layer_cfg.get("weight_decay", 0.0) or 0.0)

    def forward(self, x, prev_w):
        batch = x.shape[0]
        flat = x.view(batch, -1)
        combined = torch.cat([flat, prev_w], dim=1)
        return F.softmax(self.linear(combined), dim=1)

    def regularization_loss(self):
        if self.weight_decay:
            return self.weight_decay * torch.sum(self.linear.weight.pow(2))
        return self.linear.weight.new_tensor(0.0)


def _compute_conv_dim(size, kernel, stride, padding):
    return math.floor((size + 2 * padding - kernel) / stride) + 1


class CNN(nn.Module):
    """PyTorch implementation of the original TensorFlow CNN builder."""

    def __init__(self, feature_number, rows, columns, layers, device="cpu"):
        super().__init__()
        self.feature_number = feature_number
        self.rows = rows
        self.columns = columns
        self.layers_config = layers
        self.layer_ops = nn.ModuleList()
        self.device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
        self._build_network()
        self.to(self.device)

    def _build_network(self):
        current_channels = self.feature_number
        current_height = self.rows
        current_width = self.columns
        for layer_cfg in self.layers_config:
            layer_type = layer_cfg["type"]
            if layer_type == "ConvLayer":
                op = ConvLayerOp(current_channels, layer_cfg)
                kernel = tuple(allint(layer_cfg.get("filter_shape", [1, 1])))
                strides = tuple(allint(layer_cfg.get("strides", [1, 1])))
                padding_type = layer_cfg.get("padding", "valid").lower()
                padding = (kernel[0] // 2 if padding_type == "same" else 0,
                           kernel[1] // 2 if padding_type == "same" else 0)
                current_height = _compute_conv_dim(current_height, kernel[0], strides[0], padding[0])
                current_width = _compute_conv_dim(current_width, kernel[1], strides[1], padding[1])
                current_channels = int(layer_cfg["filter_number"])
            elif layer_type == "DropOut":
                op = DropoutOp(layer_cfg.get("keep_probability", 1.0))
            elif layer_type == "MaxPooling":
                op = PoolingOp("max", layer_cfg)
                kernel = tuple(allint(layer_cfg.get("strides", [2, 2])))
                current_height = _compute_conv_dim(current_height, kernel[0], kernel[0], 0)
                current_width = _compute_conv_dim(current_width, kernel[1], kernel[1], 0)
            elif layer_type == "AveragePooling":
                op = PoolingOp("avg", layer_cfg)
                kernel = tuple(allint(layer_cfg.get("strides", [2, 2])))
                current_height = _compute_conv_dim(current_height, kernel[0], kernel[0], 0)
                current_width = _compute_conv_dim(current_width, kernel[1], kernel[1], 0)
            elif layer_type == "LocalResponseNormalization":
                op = LRNOp()
            elif layer_type == "DenseLayer":
                in_features = current_channels * current_height * current_width
                op = DenseLayerOp(in_features, layer_cfg)
                current_channels = int(layer_cfg["neuron_number"])
                current_height = 1
                current_width = 1
            elif layer_type == "EIIE_Dense":
                op = EIIEDenseOp(current_channels, current_width, layer_cfg)
                current_channels = int(layer_cfg["filter_number"])
                current_width = 1
            elif layer_type == "EIIE_Output":
                op = EIIEOutputOp(current_channels, self.rows, current_width, layer_cfg)
            elif layer_type == "EIIE_Output_WithW":
                op = EIIEOutputWithWOp(current_channels, self.rows, current_width, layer_cfg)
            elif layer_type == "Output_WithW":
                in_features = current_channels * current_height * current_width
                op = OutputWithWOp(in_features, self.rows, layer_cfg)
            else:
                raise ValueError("Layer {} not supported in PyTorch backend.".format(layer_type))
            self.layer_ops.append(op)

    def forward(self, x, previous_w):
        # x shape: [batch, features, rows, columns]
        reference = x[:, 0:1, :, -1:]
        x = x / (reference + 1e-8)
        output = x
        for op in self.layer_ops:
            output = op(output, previous_w)
        return output

    def regularization_loss(self):
        losses: List[torch.Tensor] = []
        for op in self.layer_ops:
            reg = getattr(op, "regularization_loss", None)
            if callable(reg):
                loss = reg()
                if loss is not None:
                    losses.append(loss)
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).sum()


def allint(l):
    return [int(i) for i in l]
