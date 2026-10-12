"""Train the MNIST linear probes on the features of the integer golden model.

Usage: .venv/bin/python host/train_probe.py [--reuse]     (needs torch, timm, torchvision)
Output: host/probe_mnist112.npz

Features. Every digit goes through the integer 2-layer pass (the model the
hardware runs) up to the final LayerNorm, which gives an INT8 50 x 192 tensor.
Two probes are trained on it, each a 192 x 10 linear layer:
  mean    on the mean of the 49 patch rows
  class   on the class row (row 0)
The mean probe is linear, so the hardware applies it to every row and the host
sums the 49 score rows of the patches; the class probe is applied to row 0.

The probes are then quantized the way every weight is (INT8 weights with one
scale, INT32 bias, INT8 scores) and the accuracy is measured again with that
integer arithmetic on test digits the training never saw.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HOST_DIR), "golden"))
import gen_deit_forward as golden  # noqa: E402
import mnist_config as config  # noqa: E402
from int8_ops import DATA_MAX  # noqa: E402


def features(parameters, dataset, indices):
    """Final LayerNorm output of the integer pass, as real numbers: (digits, 50, 192)."""
    out = np.zeros((len(indices), config.TOKENS, 192))
    for position, index in enumerate(indices):
        forward = golden.Forward(parameters, golden.INTEGER_OPS).run(config.integer_input(
            config.pixels_of(dataset, index)))
        tensor, scale = next((t, s) for name, t, s, _key in forward.stages if name == "final_ln")
        assert tensor.shape == out[position].shape
        out[position] = tensor * scale
    return out


def train(train_x, train_y):
    """Multinomial logistic regression; returns weight (192 x 10) and bias in feature units."""
    mean, deviation = train_x.mean(0), train_x.std(0) + 1e-6
    x = torch.from_numpy((train_x - mean) / deviation)
    y = torch.from_numpy(train_y)
    weight = torch.zeros(x.shape[1], 10, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(10, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=500, history_size=20,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(x @ weight + bias, y) + 1e-3 * (weight ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    # fold the standardisation into the layer
    folded = weight.detach().numpy() / deviation[:, None]
    return folded, bias.detach().numpy() - mean @ folded


def accuracy(scores, labels):
    return 100.0 * float((scores.argmax(1) == labels).mean())


def main():
    torch.manual_seed(0)
    model = config.load_model()
    train_set, test_set = config.datasets()
    target = config.calibrate(model, train_set, golden.capture_all)
    parameters = golden.Model(model, target, classifiers=[], final_all_rows=True)

    # the integer pass over 8,000 digits takes a while: --reuse takes the features of the last run
    cache = os.path.join(os.path.dirname(HOST_DIR), "sim", "probe_features.npz")
    if "--reuse" in sys.argv and os.path.exists(cache):
        train_x, test_x = np.load(cache)["train"], np.load(cache)["test"]
    else:
        train_x = features(parameters, train_set, config.PROBE_TRAIN)
        test_x = features(parameters, test_set, config.EVALUATION)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, train=train_x.astype(np.float32), test=test_x.astype(np.float32))
        train_x, test_x = np.load(cache)["train"].astype(np.float64), np.load(cache)["test"].astype(np.float64)
    train_y = train_set.targets[list(config.PROBE_TRAIN)].numpy()
    test_y = test_set.targets[list(config.EVALUATION)].numpy()

    mean_weight, mean_bias = train(train_x[:, 1:].mean(1), train_y)
    class_weight, class_bias = train(train_x[:, 0], train_y)
    print(f"probes trained on {len(train_y)} digits, evaluated on {len(test_y)} others; features from the integer "
          f"model")
    print(f"  float probe:      mean of patch rows {accuracy(test_x[:, 1:].mean(1) @ mean_weight + mean_bias, test_y):.1f}%"
          f"   class row {accuracy(test_x[:, 0] @ class_weight + class_bias, test_y):.1f}%")

    # output scales: the largest score any row produces on the training digits
    mean_scale = float(np.abs(train_x @ mean_weight + mean_bias).max()) / DATA_MAX
    class_scale = float(np.abs(train_x[:, 0] @ class_weight + class_bias).max()) / DATA_MAX
    np.savez(config.PROBE_FILE, mean_weight=mean_weight, mean_bias=mean_bias, class_weight=class_weight,
             class_bias=class_bias, mean_scale=mean_scale, class_scale=class_scale)

    # the same probes in the engine's integer arithmetic
    classifiers, scales = config.load_probe()
    quantized = golden.Model(model, dict(target, **scales), classifiers=classifiers)
    final_scale = target["head_in"]
    mean_right = class_right = 0
    for features_real, label in zip(test_x, test_y):
        final = np.rint(features_real / final_scale).astype(np.int64)
        forward = golden.Forward(quantized, golden.INTEGER_OPS)
        rows, _ = forward.linear("head_mean", "head", "final_ln", final, final_scale, "head_mean_out")
        row0, _ = forward.linear("head_class", "head", "final_ln", final[0:1], final_scale, "head_class_out")
        mean_right += int(np.argmax(rows[1:].sum(0))) == label        # the host sums the patch rows
        class_right += int(np.argmax(row0[0])) == label
    print(f"  quantized (INT8): mean of patch rows {100.0 * mean_right / len(test_y):.1f}%"
          f"   class row {100.0 * class_right / len(test_y):.1f}%")
    print(f"wrote {config.PROBE_FILE}")


if __name__ == "__main__":
    main()
