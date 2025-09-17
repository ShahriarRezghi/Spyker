# Spyker

Spyker is a high-performance software stack for spiking neural networks (SNNs). It couples a
hand-optimised C++/CUDA core with ergonomic Python bindings so you can prototype new learning rules
in a notebook, benchmark large-scale models on a workstation, or integrate with production
pipelines without rewriting code. Spyker targets researchers and engineers who need biologically
plausible SNN components (filters, plasticity rules, sparse encodings) delivered with the speed and
modularity normally associated with contemporary deep-learning libraries.

> **TL;DR** Spyker brings STDP-capable, event-driven networks to commodity CPUs and GPUs with an API
> that mirrors familiar PyTorch/NumPy workflows while offering the raw performance and control of a
> native C++ engine.

## Why Spyker?

Insights from the accompanying research paper highlight why Spyker differs from existing SNN tools:

- **Performance built in.** The C++/CUDA backend was written from scratch to minimise memory
  overheads and kernel launches, yielding multi-fold speedups over PyTorch-based SNN libraries in
  image-recognition benchmarks. Batched simulations that previously required hours now complete in
  minutes on commodity GPUs.
- **Biologically grounded learning.** Out-of-the-box implementations of spike-timing-dependent
  plasticity (STDP) and reward-modulated STDP (R-STDP) let you reproduce neuroscience experiments
  or explore new local learning signals. The modular design makes it straightforward to add further
  rules.
- **Dense and sparse paths.** Spyker supports dense tensors for conventional convolutional layers as
  well as sparse spike streams with CPU-side operators—use whichever representation best matches
  your workload.
- **Scientific validation.** Representations learned by Spyker-based networks correlate strongly
  with macaque electrophysiology recordings and compare favourably to deep CNN baselines, showing
  that the library is suitable for computational neuroscience as well as machine learning.
- **Pragmatic API.** Python and C++ interfaces are mirrored: once you design a pipeline in Python,
  the equivalent C++ calls share the same names and semantics for deployment.

## Key Features

- **Layer modules:** Difference-of-Gaussians, Gabor, Laplacian-of-Gaussian filters, ZCA whitening,
  convolution, fully-connected layers, pooling, inhibition, spike coding/decoding, WTA selection,
  temporal backprop utilities, and more.
- **Plasticity toolkit:** Configurable STDP/STDP variants, integrate-and-fire spike generation,
  event aggregation, and helper data structures for winner tracking.
- **Device/runtime control:** Query and manage CUDA devices, memory pools, caching behaviour, and
  threading from either language binding.
- **Interop helpers:** Zero-copy conversion between Spyker tensors, NumPy arrays, and PyTorch
  tensors, plus convenience wrappers for image/MNIST/CSV I/O.
- **Documentation & examples:** A Sphinx site, comprehensive installation notes, Jupyter notebooks,
  and MNIST SNN reference implementations.

## Installation Overview

Spyker builds with CMake ≥ 3.24 and a C++11 compiler. Optional backends include CUDA, cuDNN,
oneDNN (DNNL), and BLAS/MKL. Two complementary guides are provided:

- [INSTALLATION.rst](./INSTALLATION.rst) – exhaustive reference of build targets, environment
  variables, and dependency discovery.
- [docs/install.rst](./docs/install.rst) – step-by-step walkthrough aimed at new users.

Typical workflows:

```bash
# CPU-only C++ build
git clone --recursive https://github.com/ShahriarRezghi/Spyker.git
cmake -S Spyker -B build -DCMAKE_BUILD_TYPE=Release \
  -DSPYKER_ENABLE_CUDA=OFF -DSPYKER_ENABLE_CUDNN=OFF
cmake --build build -j$(nproc)

# Python wheel (automatically compiles the native extension)
python -m pip install .

# Editable Python install for development
python -m pip install --editable .
```

Consult the installation guides for advanced options such as forcing MKL, selecting CUDA
architectures, or cross-compiling.

## Quickstart Examples

### Python
```python
import numpy as np
import spyker

# Pick an execution device
device = spyker.device("cuda") if spyker.cuda_available() else spyker.device("cpu")

conv = spyker.Conv(insize=1, outsize=8, kernel=(5, 5), stride=2, pad=2, device=device)
conv.stdpconfig = [spyker.STDPConfig(positive=0.01, negative=0.0075)]

fc = spyker.FC(8 * 14 * 14, 10, device=device)
fc.stdpconfig = [spyker.STDPConfig(0.02, 0.015)]

image = np.random.rand(1, 1, 28, 28).astype(np.float32)
coded = spyker.code(image, time=10, dtype="u8")
spikes = spyker.fire(coded, threshold=5.0)

potentials = conv(spikes)
winners = spyker.convwta(potentials, radius=2, count=4)
conv.stdp(spikes, winners, potentials)

logits = fc(potentials.reshape(potentials.shape[0], -1))
label = spyker.labelize(logits, threshold=0.2)
print("Predicted label:", label)
```

### C++
```cpp
#include <spyker/spyker.h>

int main()
{
    using namespace Spyker;

    Device device(Kind::CUDA, 0);
    Conv conv(device, /*in=*/1, /*out=*/16, Expand2(5, 5), Expand2(2, 2), Expand4(2));
    conv.stdpconfig.push_back(STDPConfig(0.01, 0.008));

    Tensor input(Type::F32, {1, 1, 28, 28});
    input.fill(Scalar(0.0f));

    Tensor output = conv(input);
    Winners winners = convwta(output, Expand2(2), /*count=*/4);
    conv.stdp(input, winners, output);
}
```

Run the sanity check after installation:
```bash
python - <<'PY'
import spyker
print("Spyker", spyker.version())
print("Devices:", spyker.all_devices())
PY
```

## Repository Layout
```
3rd/         # Third-party components (BLASW, oneDNN, pybind11, stb, ...)
docs/        # Sphinx documentation sources
examples/    # Reference SNN pipelines (MNIST, etc.)
src/         # C++ core, CUDA kernels, and binding glue
src/python/  # Python package modules and utilities
```

## Learning Resources
- [docs/usage.rst](./docs/usage.rst) – in-depth tour of the Python and C++ APIs, dense vs.
  sparse workflows, and best practices.
- [docs/api.rst](./docs/api.rst) – entry point for generated API reference (C++ + Python).
- [Tutorial notebooks](./tutorials) – interactive walkthroughs covering PyTorch/NumPy interop,
  sparse processing, rate coding, and more.
- [Examples](./examples) – ready-to-run MNIST SNNs (place the MNIST ubyte files in `examples/MNIST`).
- Research article ([PLOS.pdf](./PLOS.pdf)) – background, benchmarks, and neuroscientific validation
  of Spyker.

## Roadmap & Research Directions
Findings from the paper suggest several future extensions:

- Broaden the learning suite beyond STDP/R-STDP (e.g., dopamine-modulated rules, surrogate-gradient
  hybrids).
- Expand sparse CUDA support and explore neuromorphic/embedded backends.
- Add a wider catalogue of neuron and synapse models (adaptive exponential, Hodgkin–Huxley, mixed
  coding schemes).
- Investigate recurrent SNNs and event-based sensor workloads such as speech or DVS streams.

Community contributions in any of these areas are highly encouraged.

## Troubleshooting Tips
- Use `cmake -DCMAKE_FIND_DEBUG_MODE=ON` to trace missing dependencies.
- Clear artefacts (`rm -rf build dist *.egg-info`) before retrying stubborn builds.
- Ensure `CUDA_PATH` / cuDNN variables point to matching toolkit versions.
- Disable optional backends (`SPYKER_ENABLE_*`) if you only require CPU execution.

## Contributing
Bug reports, feature proposals, and pull requests are welcome. Open an issue on the
[GitHub tracker](https://github.com/ShahriarRezghi/Spyker/issues) to discuss ideas. Documentation
improvements, new tutorials, or backend integrations are especially appreciated.

## License
Spyker is released under the [BSD 3-Clause License](./LICENSE).
