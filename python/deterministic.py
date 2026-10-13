"""Makes a generator run the same way every time. Import it before numpy and torch.

One thread for every numeric library (the order of a parallel sum changes its
last bits), fixed seeds, and PyTorch's deterministic algorithms. setup() does
the PyTorch part; call it once at the start of main().
"""

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"


def setup():
    import random

    import numpy as np
    import torch

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass                      # already set, or parallel work has started
    torch.use_deterministic_algorithms(True)
