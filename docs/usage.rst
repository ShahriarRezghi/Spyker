Spyker Usage Guide
==================

.. contents::
   :local:
   :depth: 2

Introduction
------------

Spyker is a high-performance toolkit for building spiking neural network (SNN) pipelines.
It offers a modern C++ core with first-class Python bindings that integrate tightly with
NumPy, PyTorch, and the library's own zero-copy tensor types. This guide walks through the
main concepts and the complete public surface in both languages so you can prototype in
Python, ship in C++, or mix the two.

Spyker focuses on three broad use cases:

- Pre-processing with classical filters (Difference-of-Gaussians, Gabor, Laplacian-of-Gaussian) and whitening.
- Dense spiking layers with convolutional/fully connected operators, STDP learning, and temporal coding utilities.
- High-throughput sparse operators compatible with hardware-friendly spike streams.

Installation at a Glance
------------------------

Full build instructions live in ``docs/install.rst`` and on the hosted documentation, but the
key points are:

- **Python**: Build and install the ``spyker`` wheel with ``pip install .`` (PyPI packages are
  coming soon). The bindings expose everything under ``spyker`` after installation.
- **C++**: Build the shared library (``libspyker``) and add ``include/spyker`` to your include
  path. Link against the library and include ``<spyker/spyker.h>`` in translation units.
- **Dependencies**: CUDA, cuDNN, and OpenMP are optional but accelerate many operators. Spyker
  gracefully falls back to CPU implementations when accelerators are unavailable.

Core Concepts
-------------

Devices and Runtime Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spyker abstracts execution and allocation devices through ``spyker.device`` in Python and the
``Spyker::Device`` class in C++. Important helpers:

- Enumerate devices with ``spyker.all_devices()`` or ``Spyker::allDevices()``.
- Query and set CUDA state: ``cuda_available()``, ``cuda_device_count()``, ``cuda_set_device()``,
  ``cuda_current_device()``, ``cuda_memory_total()/free()/taken()/used()``.
- Manage CUDA caching: ``cuda_cache_enable()``, ``cuda_cache_clear()``, ``cuda_cache_print()``.
- Tune threading with ``spyker.max_threads()`` and ``spyker.set_max_threads()`` (mirrors
  ``Spyker::maxThreads``/``Spyker::setMaxThreads``).

Data Containers
~~~~~~~~~~~~~~~

Spyker provides shared tensor implementations that map seamlessly to NumPy/PyTorch when
possible.

- **Python** exposes ``spyker.Tensor`` and ``spyker.SparseTensor`` from the core plugin.
  Wrappers in ``spyker.utils`` perform zero-copy conversion (``wrap_array``, ``create_tensor``)
  and ecosystem-aware allocation (``create_array``, ``clone_array``, ``copy_array``).
- **C++** mirrors the same concepts with ``Spyker::Tensor`` and ``Spyker::SparseTensor``.
  Both support device/dtype conversion (``tensor.to(...)``), reshaping, deep copies, and
  hold-on-data semantics via ``Tensor::hold``.
- ``spyker.utils.CodingType`` enumerates supported spike encodings (``"rank"`` and ``"rate"``),
  while ``Spyker::Code`` serves the same role in C++.
- ``Spyker::Scalar`` encapsulates untyped values that flow through the API and can be cast on demand.
- Winner selection for STDP uses ``spyker.Winner``/``Spyker::Winner`` structs grouped inside
  ``List[List[Winner]]`` in Python or ``Spyker::Winners`` in C++.

Learning and Coding Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``spyker.STDPConfig`` / ``Spyker::STDPConfig`` define potentiation/depression learning rates
  (``positive``, ``negative``), stabilization flags, and weight bounds.
- ``spyker.BPConfig`` / ``Spyker::BPConfig`` hold scalar hyperparameters for backprop-based
  learners (scaling factor, learning rate, decay, regularisation).
- Coding helpers (``code``, ``fire``, ``gather``, ``scatter``) understand ``CodingType``/``Spyker::Code``
  and convert dense signals to spike trains or vice versa.

Using Spyker from Python
------------------------

Import Basics
~~~~~~~~~~~~~

The ``spyker`` package re-exports the most common classes and functions:

.. code-block:: python

   import spyker
   from spyker import Conv, FC, DoG, Gabor, LoG, ZCA
   from spyker import conv, fc, pool, fire, code, gather, scatter
   from spyker import device, Tensor, SparseTensor

   # Sparse operators live under spyker.sparse
   from spyker import sparse

Dense Modules
~~~~~~~~~~~~~

Python modules wrap their C++ equivalents and provide ergonomic parameter handling. All accept
NumPy ndarrays, PyTorch tensors, or Spyker tensors and return a result in the same ecosystem.

- ``DoG``: Difference-of-Gaussians bank built from ``DoGFilter`` definitions.
- ``Gabor``: Orientation- and frequency-selective filters assembled from ``GaborFilter`` records.
- ``LoG``: Laplacian-of-Gaussian kernels parameterised by a list of standard deviations.
- ``ZCA``: Whitening transform with ``fit``, ``__call__`` and ``split`` helpers plus
  ``save``/``load`` for persistence.
- ``Conv``: Trainable multi-channel convolution with STDP support (``stdpconfig``) and dense/sparse
  forward passes.
- ``FC``: Dense affine layer with STDP and backpropagation helpers.

Functional Operators
~~~~~~~~~~~~~~~~~~~~

Every module also has free-function counterparts for quick experiments or stateless pipelines:

- ``canny``: Run Canny edge detection on 2D/3D/4D inputs, returning ``uint8`` masks.
- ``conv`` / ``fc``: Stateless convolution or matrix multiply; accept arbitrary stride/padding
  (``expand2``/``expand4`` semantics) and optionally apply ``sign`` to weights for ``fc``.
- ``pad``: Spatial padding with constant fill values.
- ``threshold``: In-place or out-of-place thresholding with optional replacement value.
- ``quantize``: Quantise activations to a desired dtype in-place or out-of-place.
- ``code``: Temporal spike encoding over a configurable horizon and coding scheme.
- ``infinite``: Clamp non-finite values to a replacement.
- ``fire``: Integrate-and-fire spike generator producing dense spike tensors.
- ``gather`` / ``scatter``: Collapse/expand temporal axes for spike tensors.
- ``pool``: Max pooling with optional per-location firing rates.
- ``inhibit``: Local lateral inhibition that can operate in-place.
- ``fcwta`` / ``convwta``: Winner-take-all selection for dense or convolutional layouts.
- ``stdp``: Low-level access to STDP weight updates for convolution modules.
- ``backward``: Temporal backprop helper that rolls gradients backward through time.
- ``labelize``: Convert activation maps to integer class labels above a threshold.

Example: Dense Spiking Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import spyker

   device = spyker.device("cuda") if spyker.cuda_available() else spyker.device("cpu")

   conv = spyker.Conv(
       insize=1,
       outsize=8,
       kernel=(5, 5),
       stride=2,
       pad=2,
       device=device,
   )
   conv.stdpconfig = [spyker.STDPConfig(positive=0.01, negative=0.0075)]

   fc = spyker.FC(8 * 14 * 14, 10, device=device)
   fc.stdpconfig = [spyker.STDPConfig(0.02, 0.015)]

   image = np.random.rand(1, 1, 28, 28).astype(np.float32)
   coded = spyker.code(image, time=10, dtype="u8")
   spikes = spyker.fire(coded, threshold=5.0)

   potentials = conv(spikes)
   winners = spyker.convwta(potentials, radius=2, count=4)
   conv.stdp(spikes, winners, potentials)

   flattened = potentials.reshape(potentials.shape[0], -1)
   logits = fc(flattened)
   label = spyker.labelize(logits, threshold=0.2)

Sparse Workflows
~~~~~~~~~~~~~~~~

The ``spyker.sparse`` namespace mirrors many dense operations while storing spike events in
``SparseTensor`` objects to save memory and bandwidth.

- ``sparse.code`` encodes dense inputs directly into ``SparseTensor`` spike trains.
- ``sparse.conv`` applies dense kernels to sparse inputs with optional thresholding.
- ``sparse.pool`` and ``sparse.inhibit`` operate on sparse representations without densifying.
- ``sparse.gather`` converts sparse spikes back to dense frames; ``spyker.gather`` does the same
  for dense inputs.
- ``sparse.convwta`` selects winners from sparse activations.

.. code-block:: python

   dense = np.random.rand(4, 1, 28, 28).astype(np.float32)
   sparse = spyker.sparse.code(dense, time=12)
   kernels = conv.kernel  # reuse dense weights

   sparse_out = spyker.sparse.conv(sparse, kernels, threshold=0.1)
   sparse_pooled = spyker.sparse.pool(sparse_out, kernel=2, stride=2)
   gathered = spyker.sparse.gather(sparse_pooled, dtype="u8")

Utilities and Dataset Helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``spyker.utils`` collects quality-of-life utilities:

- Zero-copy bridges: ``wrap_array`` (with ``writeable`` control), ``create_tensor`` for manual
  allocations, and ``create_array``/``clone_array``/``copy_array`` for ecosystem-aware buffers.
- Format conversion: ``to_tensor`` (wrap PyTorch/NumPy into Spyker tensors), ``to_numpy`` and
  ``to_torch`` for the inverse direction, plus ``to_sparse`` for dense→sparse conversion.
- Dataset helpers: ``read_mnist`` (labels/images), ``read_image``/``write_image`` (with optional
  resizing and format selection), ``read_csv`` for lightweight CSV ingestion.

Device and Runtime Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``spyker.spyker_plugin.control`` submodule exposes knobs that map directly to C++ entry points.
Typical usage:

.. code-block:: python

   from spyker.spyker_plugin import control

   if control.cuda_available():
       control.cuda_set_device(0)
       control.cuda_cache_enable(True)
       print("Free/Used", control.cuda_memory_free(), control.cuda_memory_used())

   devices = control.all_devices()
   control.set_max_threads(8)

Using Spyker from C++
---------------------

Setting Up
~~~~~~~~~~

Include the umbrella header and link against ``libspyker``:

.. code-block:: cpp

   #include <spyker/spyker.h>

   int main() {
       Spyker::randomSeed(1234);
       auto dev = Spyker::Device(Spyker::Kind::CUDA, 0);
       // ...
   }

Core Types
~~~~~~~~~~

- ``Spyker::Type`` enumerates scalar types; query their sizes with ``Spyker::TypeSize``.
- ``Spyker::Device`` captures execution target (CPU or CUDA, with optional index) and supports
  comparisons for dispatch.
- ``Spyker::Tensor`` wraps dense memory with rich helpers (``copy``, ``to(Type)``, ``to(Device)``,
  ``reshape``, ``fill``) and shared ownership semantics.
- ``Spyker::SparseTensor`` stores sparse spike trains, can originate from dense tensors, and
  exposes ``dims``, ``numel``, ``shape`` and ``dense()`` conversions.
- ``Spyker::Scalar`` carries strongly typed scalars with runtime conversion via ``to(Type)``.
- ``Spyker::Winner`` / ``Spyker::Winners`` represent WTA selections.
- ``Spyker::Expand2``/``Spyker::Expand4`` assist with stride/padding broadcasting.
- Configuration structs ``Spyker::STDPConfig`` and ``Spyker::BPConfig`` match the Python surface.

Runtime Utilities
~~~~~~~~~~~~~~~~~

Global helpers in ``Spyker`` manage randomness, CUDA, and thread resources:

- ``randomSeed(Size seed)``
- ``cudaAvailable``, ``cudaDeviceCount``, ``cudaSetDevice``, ``cudaCurrentDevice``
- ``cudaArchList``, ``cudaDeviceArch``
- ``cudaMemoryTotal``, ``cudaMemoryFree``, ``cudaMemoryTaken``, ``cudaMemoryUsed``
- ``cudaCacheEnabled``, ``cudaCacheEnable``, ``cudaCacheClear``, ``cudaCachePrint``
- ``clearContext`` for releasing global resources
- ``cudaConvLight``, ``cudaConvHeuristic``, ``cudaConvForce`` to steer cuDNN algorithm choice
- ``allDevices``
- ``maxThreads`` / ``setMaxThreads``

Dense Modules
~~~~~~~~~~~~~

C++ classes parallel the Python modules and offer both CPU and CUDA constructors:

- ``Spyker::DoG`` / ``Spyker::Gabor`` / ``Spyker::LoG`` accept vectors of parameter structs and
  optional padding.
- ``Spyker::ZCA`` fits whitening transforms, applies them in-place or out-of-place, and exposes
  the learned ``mean`` and ``transform`` tensors.
- ``Spyker::Conv`` and ``Spyker::FC`` expose kernels, STDP configuration vectors, and backprop
  helpers. Both support dense ``Tensor`` I/O; ``Spyker::Conv`` also supports sparse forward passes.

.. code-block:: cpp

   using namespace Spyker;

   Conv conv(Device(Kind::CUDA, 0), /*in=*/1, /*out=*/16, Expand2(5, 5), Expand2(2, 2), Expand4(2));
   conv.stdpconfig.push_back(STDPConfig(0.01, 0.008));

   Tensor input(Type::F32, {1, 1, 28, 28});
   input.fill(Scalar(0.0f));

   Tensor output = conv(input);
   Winners winners = convwta(output, Expand2(2), /*count=*/4);
   conv.stdp(input, winners, output);

Free Functions
~~~~~~~~~~~~~~

The global namespace provides stateless operators mirroring the Python bindings:

- ``Tensor canny(Tensor input, Scalar low, Scalar high)``
- ``Tensor conv(Tensor input, Tensor kernel, Expand2 stride, Expand4 pad)`` and overloads with
  explicit output tensors.
- ``Tensor fc(Tensor input, Tensor kernel, bool sign = false)``
- ``Tensor pad(Tensor input, Expand4 pad, Scalar value = 0)``
- ``Tensor threshold(Tensor input, Scalar threshold, Scalar value = 0, bool inplace = true)``
- ``Tensor quantize(Tensor input, Type type, Scalar scale = 1, Scalar shift = 0, bool inplace = true)``
- ``Tensor code(Tensor input, Size time, bool sort, Code code)``
- ``Tensor infinite(Tensor input, Scalar value = 0, bool inplace = true)``
- ``Tensor fire(Tensor input, Scalar threshold, Type type, Code code)``
- ``Tensor gather(Tensor input, Scalar threshold, Code code)``
- ``Tensor scatter(Tensor input, Size time, Type type)``
- ``Tensor pool(Tensor input, Expand2 kernel, Expand2 stride, Expand4 pad, Tensor rates)``
- ``Tensor inhibit(Tensor input, Scalar threshold, bool inplace)``
- ``Winners fcwta(Tensor input, Size radius, Size count, Scalar threshold)``
- ``Winners convwta(Tensor input, Expand2 radius, Size count, Scalar threshold)``
- ``Tensor backward(Tensor input, Tensor target, Size time, Scalar gamma)`` and the overload with
  explicit output tensor.
- ``Tensor labelize(Tensor input, Scalar threshold)`` plus overload accepting an output tensor.

Sparse Namespace
~~~~~~~~~~~~~~~~

``Spyker::Sparse`` mirrors dense functionality while staying in the spike domain:

- ``Sparse::code`` converts dense tensors into sparse spike trains.
- ``Sparse::conv`` applies dense kernels to sparse inputs with configurable stride/padding and
  firing threshold.
- ``Sparse::pad`` adds spatial padding.
- ``Sparse::gather`` collapses sparse spikes into dense tensors (with optional preallocated output).
- ``Sparse::pool`` performs max pooling on sparse activations.
- ``Sparse::inhibit`` applies sparse lateral inhibition.
- ``Sparse::convwta`` selects winners from sparse convolutional maps.

Helper Utilities
~~~~~~~~~~~~~~~~

The ``Spyker::Helper`` namespace offers lightweight I/O helpers:

- ``Helper::CSV`` for streaming CSV parsing with configurable delimiters.
- ``Helper::readImage`` / ``Helper::writeImage`` for simple image I/O with resizing and format
  conversion.
- ``Helper::mnistData`` / ``Helper::mnistLabel`` for reading the binary MNIST dataset.

Design Notes and Best Practices
-------------------------------

Performance Tips
~~~~~~~~~~~~~~~~

- Prefer zero-copy conversions with ``wrap_array``/``to_tensor`` when bridging to PyTorch or
  NumPy. Ensure arrays are contiguous and writable when operating in-place.
- Tune CUDA caching with ``cuda_cache_enable`` and ``cuda_cache_clear`` when experimenting with
  large batch sizes to avoid fragmentation.
- Use ``Spyker::Expand2``/``Expand4`` (or their Python counterparts) to express strides/padding
  succinctly without losing intent, especially when you need asymmetric padding.

Training Patterns
~~~~~~~~~~~~~~~~~

- Maintain ``STDPConfig`` lists per layer and pass ``Winner`` selections from WTA helpers to
  ``stdp`` updates. For dense STDP, accumulate winners across batches before applying updates.
- Use ``code``/``fire`` for rank coding pipelines and ``gather`` to recover potentials for
  classification layers.
- Combine ``threshold`` + ``inhibit`` to enforce sparsity before invoking WTA and STDP.

Debugging and Validation
~~~~~~~~~~~~~~~~~~~~~~~~

- Inspect tensor metadata with ``print(tensor.shape(), tensor.type(), tensor.device())`` in C++ or
  ``tensor.shape``, ``tensor.dtype`` in Python to ensure interop conversions keep the expected
  layout.
- Generate synthetic data with NumPy/torch to unit test pipelines before connecting real sensors
  or datasets.
- Use ``cuda_cache_print`` and ``clear_context`` when diagnosing resource leaks across iterative
  experiments.

Putting It Together
-------------------

Spyker's Python and C++ APIs intentionally mirror one another. Prototype quickly in Python using
NumPy or PyTorch tensors, then port the same sequence of operations to C++ by including
``<spyker/spyker.h>`` and substituting the equivalent functions/classes. Sparse operators let you
scale to large temporal horizons without prohibitive memory usage, while dense operators and
helpers cover everything from feature extraction to final classification.

Refer back to this guide as a top-level map of the available functionality, and dive into the
inline docstrings / header comments for parameter-level detail whenever you wire new components
into your spiking pipeline.
