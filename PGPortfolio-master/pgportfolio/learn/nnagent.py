from __future__ import absolute_import, division, print_function

import os
from typing import Iterable

import numpy as np
import torch

from pgportfolio.constants import LAMBDA
import pgportfolio.learn.network as network


class NNAgent:
    """PyTorch implementation of the original TensorFlow NNAgent."""

    def __init__(self, config, restore_dir=None, device="cpu"):
        self.__config = config
        self.__coin_number = config["input"]["coin_number"]
        self.__device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
        self.__pin_memory = bool(config["training"].get("pin_memory", False))
        if self.__device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.__net = network.CNN(config["input"]["feature_number"],
                                 self.__coin_number,
                                 config["input"]["window_size"],
                                 config["layers"],
                                 device=device)
        self.__commission_ratio = self.__config["trading"]["trading_consumption"]
        self.__loss_name = self.__config["training"]["loss_function"]
        # encourage deviation from strict equal-weight when configured
        self.__anti_equal_coeff = float(self.__config["training"].get("anti_equal_coeff", 0.0) or 0.0)
        self.__global_step = 0
        self.__optimizer = self._init_optimizer()
        self.__scheduler = self._init_scheduler()
        if restore_dir:
            self.load_model(restore_dir)

    def _init_optimizer(self):
        lr = self.__config["training"]["learning_rate"]
        method = self.__config["training"]["training_method"]
        params = self.__net.parameters()
        if method == "GradientDescent":
            return torch.optim.SGD(params, lr=lr)
        if method == "RMSProp":
            return torch.optim.RMSprop(params, lr=lr)
        if method == "Adam":
            return torch.optim.Adam(params, lr=lr)
        raise ValueError("Unsupported optimizer {}".format(method))

    def _init_scheduler(self):
        decay_steps = self.__config["training"].get("decay_steps", 1)
        decay_rate = self.__config["training"].get("decay_rate", 1.0)
        if decay_rate == 1.0:
            return None

        def lr_lambda(step):
            if decay_steps <= 0:
                return 1.0
            return decay_rate ** (step / float(decay_steps))

        return torch.optim.lr_scheduler.LambdaLR(self.__optimizer, lr_lambda=lr_lambda)

    def recycle(self):
        """Release GPU memory between runs."""
        if self.__device.type == "cuda":
            torch.cuda.empty_cache()

    def _pure_pc(self, future_omega, portfolio):
        c = self.__commission_ratio
        if future_omega.shape[0] <= 1:
            return torch.zeros(0, device=self.__device)
        w_t = future_omega[:-1, 1:]
        w_t1 = portfolio[1:, 1:]
        mu = 1 - torch.sum(torch.abs(w_t1 - w_t), dim=1) * c
        return mu

    def _loss(self, weights, future_price, pv_vector, pv_without_fee, last_w_tensor):
        eps = 1e-12
        method = self.__loss_name
        base = torch.sum(weights * future_price, dim=1)
        penalty = LAMBDA * torch.mean(torch.sum(-torch.log(torch.clamp(1 + 1e-6 - weights, min=eps)), dim=1))
        if method == "loss_function4":
            return -torch.mean(torch.log(base + eps))
        if method == "loss_function5":
            return -torch.mean(torch.log(base + eps)) + penalty
        if method == "loss_function6":
            return -torch.mean(torch.log(pv_vector + eps))
        if method == "loss_function7":
            return -torch.mean(torch.log(pv_vector + eps)) + penalty
        if method == "loss_function8":
            transaction = torch.sum(torch.abs(weights[:, 1:] - last_w_tensor) * self.__commission_ratio, dim=1)
            adjusted = torch.clamp(base - transaction, min=eps)
            return -torch.mean(torch.log(adjusted))
        if method == "loss_function9":
            # 基于 loss_function5，再加一项“远离等权”的奖励
            base_loss = -torch.mean(torch.log(base + eps)) + penalty
            if self.__anti_equal_coeff > 0.0:
                # 只对非 BTC 权重施加“反等权”约束
                coin_w = weights[:, 1:]
                n = coin_w.shape[1]
                if n > 0:
                    equal_w = 1.0 / float(n)
                    diff = coin_w - equal_w
                    # 平均二范数越大 → 越远离等权，在目标里给负号变成鼓励项
                    anti_equal = torch.mean(torch.sum(diff * diff, dim=1))
                    base_loss = base_loss - self.__anti_equal_coeff * anti_equal
            return base_loss
        # default to loss_function5 for backward compatibility
        return -torch.mean(torch.log(base + eps)) + penalty

    def _forward_metrics(self, x_tensor, y_tensor, last_w_tensor):
        weights = self.__net(x_tensor, last_w_tensor)
        ones = torch.ones((weights.size(0), 1), device=self.__device)
        future_price = torch.cat([ones, y_tensor[:, 0, :]], dim=1)
        denom = torch.sum(future_price * weights, dim=1, keepdim=True) + 1e-12
        future_omega = (future_price * weights) / denom
        pv_without_fee = torch.sum(weights * future_price, dim=1)
        mu = self._pure_pc(future_omega, weights)
        if pv_without_fee.shape[0] > 1:
            factors = torch.cat([torch.ones(1, device=self.__device), mu], dim=0)
        else:
            factors = torch.ones_like(pv_without_fee)
        pv_vector = pv_without_fee * factors
        log_mean = torch.mean(torch.log(torch.clamp(pv_vector, min=1e-12)))
        log_mean_free = torch.mean(torch.log(torch.clamp(pv_without_fee, min=1e-12)))
        mean = torch.mean(pv_vector)
        std = torch.sqrt(torch.mean((pv_vector - mean) ** 2) + 1e-12)
        sharp = (mean - 1) / (std + 1e-12)
        loss = self._loss(weights, future_price, pv_vector, pv_without_fee, last_w_tensor)
        reg_loss = self.__net.regularization_loss()
        total_loss = loss + reg_loss
        return {
            "weights": weights,
            "future_price": future_price,
            "future_omega": future_omega,
            "pv_vector": pv_vector,
            "portfolio_value": torch.prod(pv_vector),
            "log_mean": log_mean,
            "log_mean_free": log_mean_free,
            "standard_deviation": std,
            "sharp_ratio": sharp,
            "loss": total_loss,
            "base_loss": loss,
            "pv_without_fee": pv_without_fee
        }

    def _to_tensor(self, array):
        tensor = torch.as_tensor(array, dtype=torch.float32)
        if self.__device.type == "cuda":
            if self.__pin_memory and tensor.device.type == "cpu":
                tensor = tensor.pin_memory()
            return tensor.to(self.__device, non_blocking=self.__pin_memory)
        return tensor

    def _ensure_last_w(self, last_w):
        tensor = self._to_tensor(last_w)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def train(self, x, y, last_w, setw):
        self.__net.train()
        x_tensor = self._to_tensor(x)
        y_tensor = self._to_tensor(y)
        last_w_tensor = self._to_tensor(last_w)
        self.__optimizer.zero_grad()
        metrics = self._forward_metrics(x_tensor, y_tensor, last_w_tensor)
        metrics["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.__net.parameters(), max_norm=5.0)
        self.__optimizer.step()
        self.__global_step += 1
        if self.__scheduler:
            self.__scheduler.step()
        if setw:
            setw(metrics["weights"][:, 1:].detach().cpu().numpy())
        return metrics

    def evaluate_tensors(self, x, y, last_w, setw, tensors: Iterable[str]):
        self.__net.eval()
        x_tensor = self._to_tensor(x)
        y_tensor = self._to_tensor(y)
        last_w_tensor = self._to_tensor(last_w)
        with torch.no_grad():
            metrics = self._forward_metrics(x_tensor, y_tensor, last_w_tensor)
        if setw:
            setw(metrics["weights"][:, 1:].cpu().numpy())
        requested = []
        for key in tensors:
            if key not in metrics:
                raise KeyError("Metric {} is not available.".format(key))
            value = metrics[key]
            if isinstance(value, torch.Tensor):
                requested.append(value.detach().cpu().numpy())
            else:
                requested.append(value)
        return requested

    def save_model(self, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        payload = {
            "model_state": self.__net.state_dict(),
            "optimizer_state": self.__optimizer.state_dict(),
            "config": self.__config,
            "global_step": self.__global_step
        }
        torch.save(payload, path)

    def load_model(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError("Model file {} not found".format(path))
        payload = torch.load(path, map_location=self.__device)
        self.__net.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            self.__optimizer.load_state_dict(payload["optimizer_state"])
        self.__global_step = payload.get("global_step", 0)

    def decide_by_history(self, history, last_w):
        assert isinstance(history, np.ndarray), \
            "the history should be a numpy array, not %s" % type(history)
        assert not np.any(np.isnan(history))
        assert not np.any(np.isnan(last_w))
        history = history[np.newaxis, :, :, :]
        last_w = last_w[np.newaxis, :]
        if last_w.shape[-1] == self.__coin_number + 1:
            last_w = last_w[:, 1:]
        last_w_tensor = self._to_tensor(last_w)
        history_tensor = self._to_tensor(history)
        self.__net.eval()
        with torch.no_grad():
            weights = self.__net(history_tensor, last_w_tensor)
        return np.squeeze(weights.cpu().numpy())
