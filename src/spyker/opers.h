// BSD 3-Clause License
//
// Copyright (c) 2022, Shahriar Rezghi <shahriar25.ss@gmail.com>,
//                     Mohammad-Reza A. Dehaqani <dehaqani@ut.ac.ir>,
//                     University of Tehran
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <spyker/utils.h>

namespace Spyker
{
/// Set the seed of the random generator of the library.
/// @param seed random seed value.
SpykerExport void randomSeed(Size seed);

/// Check if CUDA is available.
SpykerExport bool cudaAvailable();

/// Get number of available CUDA devices.
SpykerExport Size cudaDeviceCount();

/// Set the current CUDA device.
SpykerExport void cudaSetDevice(Size index);

/// Get the current CUDA device.
SpykerExport Size cudaCurrentDevice();

/// Get supported CUDA architectures.
SpykerExport std::vector<Size> cudaArchList();

/// Get current CUDA architectures.
SpykerExport std::vector<Size> cudaDeviceArch();

/// Get CUDA memory capacity for current device.
SpykerExport Size cudaMemoryTotal();

/// Get CUDA free memory for current device.
SpykerExport Size cudaMemoryFree();

/// Get CUDA cached memory for current device.
SpykerExport Size cudaMemoryTaken(Size device = -1);

/// Get CUDA used memory for current device.
SpykerExport Size cudaMemoryUsed(Size device = -1);

/// Check if CUDA cache is enabled.
SpykerExport bool cudaCacheEnabled(Size device = -1);

/// Enable/Disable CUDA memory caching.
SpykerExport void cudaCacheEnable(bool enable, Size device = -1);

/// Clear CUDA memory cache.
SpykerExport void cudaCacheClear(Size device = -1);

/// Clear all reserved data.
SpykerExport void clearContext();

/// Print a list of allocated pointers in CUDA memory cache.
SpykerExport void cudaCachePrint(Size device = -1);

/// Set light convolution mode.
///
/// When CUDNN is enabled and light convolution is active, newely created convolution descriptors
/// (not cached ones) use algorithms that take up less memory. Enabling this might decrease performance in some cases.
/// This is disabled by default.
///
/// @param light light convolution mode.
SpykerExport void cudaConvLight(bool light);

/// Set heuristic convolution algorithm finding mode.
///
/// When CUDNN is enabled and heuristic algorithm finding is active, newely created convolution descriptors
/// (not cached ones) will use heuristics to find the fastest convolution algorithms instead of running samples.
/// Enabling this might decrease performance in some cases. This is enabled by default.
///
/// @param heuristic heuristic convolution mode.
SpykerExport void cudaConvHeuristic(bool heuristic);

/// Set heuristic convolution algorithm finding mode.
///
/// When CUDNN is enabled and heuristic algorithm finding is active, newely created convolution descriptors
/// (not cached ones) will use heuristics to find the fastest convolution algorithms instead of running samples.
/// Enabling this might decrease performance in some cases. This is enabled by default.
///
/// @param heuristic heuristic convolution mode.
SpykerExport void cudaConvForce(bool force);

/// Get a list of all the available devices that the library can use.
/// @return a list of available devices.
SpykerExport std::vector<Device> allDevices();

/// Get the maximum number of threads used
/// @return maximum number of threads used
SpykerExport Size maxThreads();

/// Set the maximum number of threads
/// @param threads maximum number of threads used
SpykerExport void setMaxThreads(Size threads = -1);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// 2D shape container.
///
/// This class broadcasts shapes to 2D.
class SpykerExport Expand2
{
    Shape data;

public:
    inline Expand2() : data{0, 0} {}
    inline Expand2(Size _1) : data{_1, _1} {}
    inline Expand2(Size _1, Size _2) : data{_1, _2} {}
    inline Size operator[](Size i) const { return data[i]; }
    inline Shape get() const { return data; }
    operator Shape() { return data; }
    Expand2(const Shape &shape);
};

/// 4D shape container.
///
/// This class broadcasts shapes to 4D.
class SpykerExport Expand4
{
    Shape data;

public:
    inline Expand4() : data{0, 0, 0, 0} {}
    inline Expand4(Size _1) : data{_1, _1, _1, _1} {}
    inline Expand4(Size _1, Size _2) : data{_1, _2, _1, _2} {}
    inline Expand4(Size _1, Size _2, Size _3, Size _4) : data{_1, _2, _3, _4} {}
    inline Size operator[](Size i) const { return data[i]; }
    inline Shape get() const { return data; }
    operator Shape() { return data; }
    Expand4(const Shape &shape);
};

/// Difference of Gaussian filter parameter container.
struct SpykerExport DoGFilter
{
    /// Standard deviation of the first gaussian filter.
    F64 std1;

    /// Standard deviation of the second gaussian filter.
    F64 std2;

    /// Initialize Filter with parameters.
    inline DoGFilter(F64 std1, F64 std2) : std1(std1), std2(std2) {}
};

/// Gabor filter parameter container.
struct SpykerExport GaborFilter
{
    /// Standard deviation
    ///
    /// Width of the strips of the filter.
    F64 sigma;

    /// Orientation
    ///
    /// Orientation of the filter (unit degrees).
    F64 theta;

    /// Spatial aspect ratio
    ///
    /// Height of the stripes, reverse relation.
    F64 gamma;

    /// Wavelength
    ///
    /// Spacing between strips, reverse relation.
    F64 lambda;

    /// Phase offset
    ///
    /// Spacial shift of the strips.
    F64 psi;

    inline GaborFilter(F64 sigma, F64 theta, F64 gamma, F64 lambda, F64 psi)
        : sigma(sigma), theta(theta), gamma(gamma), lambda(lambda), psi(psi)
    {
        this->theta *= 0.017453292519943295;
    }
};

/// 2D difference of Gaussian (DoG) filter module.
class SpykerExport DoG
{
    bool _init;
    Device _device;
    Expand4 _pad;

public:
    /// Kernel of the filter.
    Tensor kernel;

    /// Filter can't be used when initialized this way.
    DoG();

    /// @param size half size of the window. full size of the window is 2 * size + 1.
    /// @param filters list of 'DoGFilter's to be applied.
    /// @param pad padding size of the input (default 0).
    DoG(Size size, const std::vector<DoGFilter> &filters, Expand4 pad = 0, Type type = Type::F32);

    /// @param device device of the filter to be run on.
    /// @param size half size of the window. full size of the window is 2 * size + 1.
    /// @param filters list of 'DoGFilter's to be applied.
    /// @param pad padding size of the input (default 0).
    DoG(Device device, Size size, const std::vector<DoGFilter> &filters, Expand4 pad = 0, Type type = Type::F32);

    /// Apply the filter on the input.
    ///
    /// @param[in] input input dense tensor to be processed.
    /// @param[out] output dense tensor to be written to.
    Tensor operator()(Tensor input, Tensor output);

    /// Apply the filter on the input.
    ///
    /// @param[in] input input dense tensor to be processed.
    /// @return filtered output dense tensor.
    Tensor operator()(Tensor input);

    /// Get the device of this module.
    Device device() const;
};

/// 2D Gabor filter module.
class SpykerExport Gabor
{
    bool _init;
    Device _device;
    Expand4 _pad;

public:
    /// Kernel of the filter.
    Tensor kernel;

    /// Filter can't be used when initialized this way.
    Gabor();

    /// @param size half size of the window. full size of the window is 2 * size + 1.
    /// @param filters list of 'GaborFilter's to be applied.
    /// @param pad padding size of the input (TBLR) (default 0).
    Gabor(Size size, const std::vector<GaborFilter> &filters, Expand4 pad = 0, Type type = Type::F32);

    /// @param device device of the filter to be run on.
    /// @param size half size of the window. full size of the window is 2 * size + 1.
    /// @param filters list of 'GaborFilter's to be applied.
    /// @param pad padding size of the input (TBLR) (default 0).
    Gabor(Device device, Size size, const std::vector<GaborFilter> &filters, Expand4 pad = 0, Type type = Type::F32);

    /// Apply the filter on the input.
    ///
    /// @param[in] input input dense tensor to be processed.
    /// @param[out] output dense tensor to be written to.
    Tensor operator()(Tensor input, Tensor output);

    /// Apply the filter on the input.
    ///
    /// @param[in] input input dense tensor to be processed.
    /// @return filtered output dense tensor.
    Tensor operator()(Tensor input);

    /// Get the device of this module.
    Device device() const;
};

/// 2D Laplacian of Gaussian (LoG) filter module.
class SpykerExport LoG
{
    bool _init;
    Device _device;
    Expand4 _pad;

public:
    /// Kernel of the filter.
    Tensor kernel;

    /// Filter can't be used when initialized this way.
    LoG();

    /// @param size half size of the window. full size of the window is 2 * size + 1.
    /// @param stds list of stds for the LoG filters to be applied.
    /// @param pad padding size of the input (TBLR) (default 0).
    LoG(Size size, const std::vector<F64> &stds, Expand4 pad = 0, Type type = Type::F32);

    /// @param device device of the filter to be run on.
    /// @param size half size of the window. full size of the window is 2 * size + 1.
    /// @param stds list of stds for the LoG filters to be applied.
    /// @param pad padding size of the input (TBLR) (default 0).
    LoG(Device device, Size size, const std::vector<F64> &stds, Expand4 pad = 0, Type type = Type::F32);

    /// Apply the filter on the input.
    ///
    /// @param[in] input input dense tensor to be processed.
    /// @param[out] output dense tensor to be written to.
    Tensor operator()(Tensor input, Tensor output);

    /// Apply the filter on the input.
    ///
    /// @param[in] input input dense tensor to be processed.
    /// @return filtered output dense tensor.
    Tensor operator()(Tensor input);

    /// Get the device of this module.
    Device device() const;
};

/// ZCA whitening transformation module.
class SpykerExport ZCA
{
public:
    /// Mean values of features.
    Tensor mean;

    /// Whitening transformation matrix.
    Tensor transform;

    /// Empty initialization.
    ZCA();

    /// Fit the input data
    /// @param[in, out] input input data to be fitted to.
    /// @param epsilon epsilon parameter of the trnasformation.
    /// @param transform whether to transform the input inplace or not (default false).
    /// @return reference to this class
    ZCA &fit(Tensor input, Scalar epsilon, bool transform = false);

    /// Transform the input data
    /// @param[in, out] input input data to be transformed.
    /// @param inplace whether to transform the input inplace or not (default true).
    /// @return transformed data
    Tensor operator()(Tensor input, bool inplace = true);

    /// Split ZCA transformed data into two positive and negative channels.
    /// @param[in] input input data to be splitted.
    /// @param[out] output output tensor.
    /// @return the output tensor.
    static Tensor split(Tensor input, Tensor output);

    /// Split ZCA transformed data into two positive and negative channels.
    /// @param[in] input input data to be splitted.
    /// @return splitted data.
    static Tensor split(Tensor input);
};

/// 2D convolution module.
class SpykerExport Conv
{
    bool _init;
    Device _device;
    Expand2 _stride;
    Expand4 _pad;

public:
    /// Kernel of the module.
    Tensor kernel;

    /// List of STDP configurations.
    std::vector<STDPConfig> stdpconfig;

    /// Module can't be used when initialized this way.
    Conv();

    /// @param input channels of the input signal.
    /// @param output channels of the output signal.
    /// @param kernel kernel size of the convolution.
    /// @param stride stride size of the convolution (default 1).
    /// @param pad padding size of the convolution (default 0).
    /// @param mean mean of the random normal variable that initializes the kernel (default 0.5).
    /// @param std standard deviation of the random normal variable that initializes the kernel (default 0.02).
    Conv(Size input, Size output, Expand2 kernel, Expand2 stride = 1, Expand4 pad = 0, F64 mean = .5, F64 std = .02,
         Type type = Type::F32);

    /// @param device device of the module to be run on.
    /// @param input channels of the input signal.
    /// @param output channels of the output signal.
    /// @param kernel kernel size of the convolution.
    /// @param stride size of the convolution (default 1).
    /// @param pad padding size of the convolution (default 0).
    /// @param mean mean of the random normal variable that initializes the kernel (default 0.5).
    /// @param std standard deviation of the random normal variable that initializes the kernel (default 0.02).
    Conv(Device device, Size input, Size output, Expand2 kernel, Expand2 stride = 1, Expand4 pad = 0, F64 mean = .5,
         F64 std = .02, Type type = Type::F32);

    /// Apply the convolution on the input.
    ///
    /// @param[in] input input dense tensor to be processed.
    /// @param[out] output dense tensor to be written to.
    Tensor operator()(Tensor input, Tensor output);

    /// Apply the convolution on the input.
    ///
    /// @param[in] input input dense tensor to be processed.
    /// @return convolved output dense tensor.
    Tensor operator()(Tensor input);

    // TODO
    /// Apply 2D Convolution
    ///
    /// @param conv convlution class to be used.
    /// @param input input sparse tensor to be convolved.
    /// @param threshold threshold to be applied to potentials.
    /// @return convolution and intergreate-and-fire output spikes.
    SparseTensor operator()(SparseTensor input, Scalar threshold);

    /// Apply the STDP on the convolution.
    ///
    /// @param[in] input convolution input dense tensor.
    /// @param winners winner neurons that are selected for updating.
    /// @param[in] output convolution output dense tensor.
    void stdp(Tensor input, const Winners &winners, Tensor output);

    // TODO
    /// Apply the STDP on the convolution.
    ///
    /// @param conv convolution class to be used.
    /// @param input convolution input sparse tensor.
    /// @param winners winner neurons that are selected for updating.
    void stdp(SparseTensor input, const Winners &winners);

    /// Get the device of this module.
    Device device() const;
};

/// Fully connected module
class SpykerExport FC
{
    bool _init;
    Device _device;

public:
    /// Kernel of the module.
    Tensor kernel;

    /// List of STDP configurations.
    std::vector<STDPConfig> stdpconfig;

    /// Backpropagation configutation.
    BPConfig bpconfig;

    /// Module can't be used when initialized this way.
    FC();

    /// @param input dimensions of the input signal.
    /// @param output dimensions of the output signal.
    /// @param mean mean of the random normal variable that initializes the kernel (default 0.5).
    /// @param std standard deviation of the random normal variable that initializes the kernel (default 0.02).
    FC(Size input, Size output, F64 mean = .5, F64 std = .02, Type type = Type::F32);

    /// @param device device of the module to be run on.
    /// @param input dimensions of the input signal.
    /// @param output dimensions of the output signal.
    /// @param mean mean of the random normal variable that initializes the kernel (default 0.5).
    /// @param std standard deviation of the random normal variable that initializes the kernel (default 0.02).
    FC(Device device, Size input, Size output, F64 mean = .5, F64 std = .02, Type type = Type::F32);

    /// Apply the fully connected on the input.
    ///
    /// @param[in] input input dense tensor to be processed.
    /// @param[out] output dense tensor to be written to.
    Tensor operator()(Tensor input, Tensor output, bool sign = false);

    /// Apply the fully connected on the input.
    ///
    /// @param[in] input input dense tensor to be processed.
    /// @return fully connected output dense tensor.
    Tensor operator()(Tensor input, bool sign = false);

    /// Backpropagate and return the gradient.
    Tensor backward(Tensor input, Tensor output, Tensor grad);

    /// Backpropagate and return the gradient.
    Tensor backward(Tensor input, Tensor output, Tensor grad, Tensor next);

    /// Apply the STDP on the fully connected.
    ///
    /// @param[in] input fully connected input dense tensor.
    /// @param winners winner neurons that are selected for updating.
    /// @param[in] output fully connected output dense tensor.
    void stdp(Tensor input, const Winners &winners, Tensor output);

    /// Get the device of this module.
    Device device() const;
};

/// Apply Canny edge detection.
///
/// @param[in] input edge detection input dense tensor.
/// @param low threshold for weak edges.
/// @param high threshold for strong edges.
/// @return edge detection output dense tensor.
SpykerExport Tensor canny(Tensor input, Scalar low, Scalar high);

/// Apply Canny edge detection.
///
/// @param[in] input edge detection input dense tensor.
/// @param[out] output edge detection output dense tensor.
/// @param low threshold for weak edges.
/// @param high threshold for strong edges.
/// @return the output tensor.
SpykerExport Tensor canny(Tensor input, Tensor output, Scalar low, Scalar high);

/// Apply 2D convolution.
///
/// @param[in] input convolution input dense tensor.
/// @param[in] kernel convolution kernel dense tensor.
/// @param stride convolution stride (default 1).
/// @param pad input padding (default 0).
/// @return convolution output dense tensor.
SpykerExport Tensor conv(Tensor input, Tensor kernel, Expand2 stride = 1, Expand4 pad = 0);

/// Apply 2D convolution.
///
/// @param[in] input convolution input dense tensor.
/// @param[in] kernel convolution kernel dense tensor.
/// @param[out] output convolution output dense tensor.
/// @param stride convolution stride (default 1).
/// @param pad input padding (default 0).
/// @return the output tensor.
SpykerExport Tensor conv(Tensor input, Tensor kernel, Tensor output, Expand2 stride = 1, Expand4 pad = 0);

/// Apply fully connected.
///
/// @param[in] input fully connected input dense tensor.
/// @param[in] kernel fully connected kernel dense tensor.
/// @return fully connected output dense tensor.
SpykerExport Tensor fc(Tensor input, Tensor kernel, bool sign = false);

/// Apply fully connected.
///
/// @param[in] input fully connected input dense tensor.
/// @param[in] kernel fully connected kernel dense tensor.
/// @param[out] output fully connected output dense tensor.
/// @return the output tensor.
SpykerExport Tensor fc(Tensor input, Tensor kernel, Tensor output, bool sign = false);

/// Apply 2D padding.
///
/// @param[in] input input dense tensor.
/// @param pad input padding (1D->S, 2D->VH, 4D->TLBR).
/// @param value padding value (default 0).
/// @return padding output dense tensor.
SpykerExport Tensor pad(Tensor input, Expand4 pad, Scalar value = 0);

/// 2D padding.
///
/// @param[in] input input dense tensor.
/// @param[out] output output dense tensor.
/// @param pad input padding (1D->S, 2D->VH, 4D->TLBR).
/// @param value padding value (default 0).
/// @return the output tensor.
SpykerExport Tensor pad(Tensor input, Tensor output, Expand4 pad, Scalar value = 0);

/// Apply thresholding.
///
/// @param[in, out] input input dense tensor.
/// @param threshold the threshold value.
/// @param value value to replace (default 0).
/// @param inplace override the input or not (default true).
/// @return if inplace input tensor, otherwise output tensor.
SpykerExport Tensor threshold(Tensor input, Scalar threshold, Scalar value = 0, bool inplace = true);

/// Quantize input.
///
/// If values are lower than middle then it is set to lower and set to upper otherwise.
///
/// @param[in, out] input input dense tensor to be quantized.
/// @param lower lower bound value.
/// @param middle middle value to be compared.
/// @param upper upper bound value.
/// @param inplace override the input or not (default true).
/// @return if inplace input tensor, otherwise output tensor.
SpykerExport Tensor quantize(Tensor input, Scalar lower, Scalar middle, Scalar upper, bool inplace = true);

/// Apply rank coding.
///
/// @param[in] input input dense tensor.
/// @param time number of time steps.
/// @param sort whether to sort the values or not.
/// Sorting might increase accuracy but it deceases performance (default true).
/// @param type data type of the created output tensor (default Type::U8).
/// @return rank coding output dense tensor.
SpykerExport Tensor code(Tensor input, Size time, bool sort = true, Type type = Type::U8, Code code = Code::Rank);

/// Apply rank coding.
///
/// @param[in] input input dense tensor.
/// @param[out] output output dense tensor.
/// @param time number of time steps.
/// @param sort whether to sort the values or not.
/// Sorting might increase accuracy but it deceases performance (default true).
/// @return the output tensor.
SpykerExport Tensor code(Tensor input, Tensor output, Size time, bool sort = true, Code code = Code::Rank);

/// Apply infinite thresholding.
///
/// @param[in, out] input input dense tensor.
/// @param value value to replace (default 0).
/// @param inplace override the input or not (default true).
/// @return if inplace input tensor, otherwise output tensor.
SpykerExport Tensor infinite(Tensor input, Scalar value = 0, bool inplace = true);

/// Apply integrate-and-fire mechanism.
///
/// If the input is already thresholded then there is no need to pass threshold to this function.
///
/// @param[in] input input dense tensor.
/// @param threshold threshold of firing (default 0).
/// @param type data type of the created output tensor (default Type::U8).
/// @return integrate-and-fire output dense tensor.
SpykerExport Tensor fire(Tensor input, Scalar threshold = 0, Type type = Type::U8, Code code = Code::Rank);

/// Apply integrate-and-fire mechanism.
///
/// If the input is already thresholded then there is no need to pass threshold to this function.
///
/// @param[in] input input dense tensor.
/// @param[out] output output dense tensor.
/// @param threshold threshold of firing (default 0).
/// @return the output tensor.
SpykerExport Tensor fire(Tensor input, Tensor output, Scalar threshold = 0, Code code = Code::Rank);

/// Gather temporal information.
///
/// If the input is already thresholded then there is no need to pass threshold to this function.
///
/// @param[in] input input dense tensor.
/// @param threshold threshold of integrate-and-fire layer (default 0).
/// @param type data type of the created output tensor (default Type::U8).
/// @return gathered output dense tensor.
SpykerExport Tensor gather(Tensor input, Scalar threshold = 0, Type type = Type::U8, Code code = Code::Rank);

/// Gather temporal information.
///
/// If the input is already thresholded then there is no need to pass threshold to this function.
///
/// @param[in] input input dense tensor.
/// @param[out] output output dense tensor.
/// @param threshold threshold of integrate-and-fire layer (default 0).
/// @return the output tensor.
SpykerExport Tensor gather(Tensor input, Tensor output, Scalar threshold = 0, Code code = Code::Rank);

/// Scatter gathered temporal information.
///
/// @param[in] input input dense tensor.
/// @param time number of time steps of the output.
/// @param type data type of the created output tensor (default Type::U8).
/// @return scattered output tensor.
SpykerExport Tensor scatter(Tensor input, Size time, Type type = Type::U8);

/// Scatter gathered temporal information.
///
/// @param[in] input input dense tensor.
/// @param[out] output output dense tensor.
/// @return the output tensor.
SpykerExport Tensor scatter(Tensor input, Tensor output);

/// Apply 2D pooling.
///
/// @param[in] input pooling input dense tensor.
/// @param[in] kernel pooling kernel dense tensor.
/// @param stride convolution stride. Zero stride means same as kernel (default 0).
/// @param pad input padding (default 0).
/// @return pooling output dense tensor.
SpykerExport Tensor pool(Tensor input, Expand2 kernel, Expand2 stride = 0, Expand4 pad = 0, Tensor rates = {});

/// Apply 2D pooling.
///
/// @param[in] input pooling input dense tensor.
/// @param[out] output pooling output dense tensor.
/// @param kernel pooling kernel size.
/// @param stride convolution stride. Zero stride means same as kernel (default 0).
/// @param pad input padding (default 0).
/// @return the output tensor.
SpykerExport Tensor pool(Tensor input, Tensor output, Expand2 kernel, Expand2 stride = 0, Expand4 pad = 0,
                         Tensor rates = {});

/// Apply lateral Inhibition (inplace).
///
/// If the input is already thresholded then there is no need to pass threshold to this function.
///
/// @param[in, out] input input dense tensor.
/// @param threshold threshold of integrate-and-fire layer (default 0).
/// @param inplace override the input or not (default true).
/// @return if inplace input tensor, otherwise output tensor.
SpykerExport Tensor inhibit(Tensor input, Scalar threshold = 0, bool inplace = true);

/// Select winner neurons from fully connected output.
///
/// If the input is already thresholded then there is no need to pass threshold to this function.
///
/// @param[in] input input dense tensor.
/// @param radius radius of inhibition.
/// @param threshold threshold of integrate-and-fire layer (default 0).
/// @param count number of neurons that will be selected.
/// @return winner neurons.
SpykerExport Winners fcwta(Tensor input, Size radius, Size count, Scalar threshold = 0);

/// Select winner neurons from convolution output.
///
/// If the input is already thresholded then there is no need to pass threshold to this function.
///
/// @param[in] input input dense tensor.
/// @param radius radius of inhibition.
/// @param count number of neurons that will be selected.
/// @param threshold threshold of integrate-and-fire layer (default 0).
/// @return winner neurons.
SpykerExport Winners convwta(Tensor input, Expand2 radius, Size count, Scalar threshold = 0);

SpykerExport Tensor backward(Tensor input, Tensor target, Size time, Scalar gamma);

SpykerExport Tensor backward(Tensor input, Tensor output, Tensor target, Size time, Scalar gamma);

SpykerExport Tensor labelize(Tensor input, Scalar threshold);

SpykerExport Tensor labelize(Tensor input, Tensor output, Scalar threshold);

namespace Sparse
{
/// Apply 2D Convolution
///
/// @param input input sparse tensor to be convolved.
/// @param kernel convolution kernel dense tensor.
/// @param threshold threshold to be applied to potentials.
/// @param stride convolution stride (default 1).
/// @param pad input padding (default 0).
/// @return convolution and intergreate-and-fire output sparse spikes.
SpykerExport SparseTensor conv(SparseTensor input, Tensor kernel, Scalar threshold, Expand2 stride = 1,
                               Expand4 pad = 0);

/// Apply 2D padding.
///
/// @param input input sparse tensor.
/// @param pad input padding (1D->S, 2D->VH, 4D->TLBR).
/// @return padding output sparse tensor.
SpykerExport SparseTensor pad(SparseTensor input, Expand4 pad);

/// Apply rank coding.
///
/// @param input input sparse tensor.
/// @param time number of time steps.
/// @param sort whether to sort the values or not.
/// Sorting might increase accuracy but it deceases performance (default true).
/// @return rank coding output sparse tensor.
SpykerExport SparseTensor code(Tensor input, Size time, bool sort = true);

/// Gather temporal information.
///
/// @param input input sparse tensor.
/// @param type data type of the created output tensor (default Type::U8).
/// @return gathered output dense tensor.
SpykerExport Tensor gather(SparseTensor input, Type type = Type::U8);

/// Gather temporal information.
///
/// @param input input sparse tensor.
/// @param[out] output output dense tensor.
/// @return the output tensor.
SpykerExport Tensor gather(SparseTensor input, Tensor output);

/// Apply 2D pooling.
///
/// @param input pooling input sparse tensor.
/// @param[in] kernel pooling kernel dense tensor.
/// @param stride convolution stride. Zero stride means same as kernel (default 0).
/// @param pad input padding (default 0).
/// @return pooling output sparse tensor.
SpykerExport SparseTensor pool(SparseTensor input, Expand2 kernel, Expand2 stride = 0, Expand4 pad = 0);

/// Apply lateral Inhibition.
///
/// @param input input sparse tensor.
/// @return output inhibites sparse tensor.
SpykerExport SparseTensor inhibit(SparseTensor input);

/// Select winner neurons from convolution output.
///
/// @param input input sparse tensor.
/// @param radius radius of inhibition.
/// @param count number of neurons that will be selected.
/// @return winner neurons.
SpykerExport Winners convwta(SparseTensor input, Expand2 radius, Size count);
}  // namespace Sparse
}  // namespace Spyker
