#include <cuda_runtime.h>
#include <glog/logging.h>

#include "hastings/libinfer/cu_tensor.h"
#include "hastings/libinfer/ops.h"

namespace libinfer {
namespace detail {
__device__ float sigmoid(const float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ void subPixelPeak(const CudaTensor<const float, Ordering::NCHW> logits, const int y, const int x, Peak& peak) {
    const auto x1 = logits(peak.n, peak.c, y, x);
    const auto y1 = x1;

    const auto x0 = logits(peak.n, peak.c, y, x - 1);
    const auto x2 = logits(peak.n, peak.c, y, x + 1);

    const auto y0 = logits(peak.n, peak.c, y - 1, x);
    const auto y2 = logits(peak.n, peak.c, y + 1, x);

    const auto epsilon = 1e-6f;
    auto dx = 0.5 * (x0 - x2) / (x0 + x2 - 2.0f * x1 + epsilon);
    auto dy = 0.5 * (y0 - y2) / (y0 + y2 - 2.0f * y1 + epsilon);

    dx = fmaxf(-0.5f, fminf(0.5, dx));
    dy = fmaxf(-0.5f, fminf(0.5, dy));
}

template <int num_c>
__global__ void preprocessKernel(const CudaTensor<const unsigned char, Ordering::NHWC> image, const Normalize<num_c> norm,
                                 CudaTensor<float, Ordering::NCHW> output) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= image.w || y >= image.h || n >= image.n) {
        return;
    }

    for (int c = 0; c < num_c; ++c) {
        const auto value = float(image(n, c, y, x)) / 255.0f;
        output(n, c, y, x) = (value - norm.mean[c]) / norm.std[c];
    }
}

__global__ void sigmoidKernel(detail::CudaTensor<float, Ordering::NCHW> logits) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= logits.w || y >= logits.h) {
        return;
    }

    for (int i = 0; i < logits.n; ++i) {
        for (int j = 0; j < logits.c; ++j) {
            logits(i, j, y, x) = sigmoid(logits(i, j, y, x));
        }
    }
}

__global__ void maskFromProbsKernel(const CudaTensor<const float, Ordering::NCHW> probs, CudaTensor<unsigned char, Ordering::NHWC> image) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= image.w || y >= image.h || n >= image.n) {
        return;
    }

    for (int c = 0; c < probs.c; ++c) {
        image(n, c, y, x) = std::uint8_t(255.0 * probs(n, c, y, x));
    }
}

__global__ void colorByProbsKernel(const CudaTensor<const float, Ordering::NCHW> probs, CudaTensor<unsigned char, Ordering::NHWC> image) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= image.w || y >= image.h || n >= image.n) {
        return;
    }

    float color[3] = {0, 0, 0};
    const unsigned char colors[6][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {0, 255, 255}};

    for (int c = 0; c < probs.c; ++c) {
        const auto prob = probs(n, c, y, x);
        const auto& selected_color = colors[c % 6];
        color[0] += prob * selected_color[0];
        color[1] += prob * selected_color[1];
        color[2] += prob * selected_color[2];
    }

    for (int c = 0; c < 3; ++c) {
        image(n, c, y, x) = 0.5f * image(n, c, y, x) + 0.5f * color[c];
    }
}

template <class Fn>
__global__ void findPeaksKernel(const CudaTensor<const float, Ordering::NCHW> logits, const float threshold, int* num_peaks,
                                const int max_num_peaks, Fn fn) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int nc = blockIdx.z * blockDim.z + threadIdx.z;

    const int n = nc / logits.c;
    const int c = nc % logits.c;

    const int max_w = logits.w - 1;
    const int max_h = logits.h - 1;

    if (x >= max_w || y >= max_h) {
        return;
    }

    const auto value = logits(n, c, y, x);

    bool is_peak = value >= threshold;
    is_peak &= logits(n, c, y - 1, x - 1) < value;
    is_peak &= logits(n, c, y - 1, x) < value;
    is_peak &= logits(n, c, y - 1, x + 1) < value;

    is_peak &= logits(n, c, y, x - 1) < value;
    is_peak &= logits(n, c, y, x + 1) < value;

    is_peak &= logits(n, c, y + 1, x - 1) < value;
    is_peak &= logits(n, c, y + 1, x) < value;
    is_peak &= logits(n, c, y + 1, x + 1) < value;

    if (is_peak) {
        const int peak_idx = atomicAdd(num_peaks, 1);

        if (peak_idx < max_num_peaks) {
            fn(peak_idx, n, c, y, x, value);
        }
    }
}

struct PeakHelper {
    detail::CudaTensor<const float, Ordering::NCHW> cu_logits;
    Peak* cu_peaks = nullptr;

    __device__ void operator()(const int peak_idx, const int n, const int c, const int y, const int x,
                               const float value) __restrict__ const {
        Peak peak = {n, c, float(x), float(y), value};
        detail::subPixelPeak(cu_logits, y, x, peak);
        cu_peaks[peak_idx] = peak;
    }
};

struct OAPHelper {
    detail::CudaTensor<const float, Ordering::NCHW> cu_logits;
    detail::CudaTensor<const float, Ordering::NCHW> cu_shapes;
    Detection* cu_detections = nullptr;

    __device__ void operator()(const int peak_idx, const int n, const int c, const int y, const int x,
                               const float value) __restrict__ const {
        const auto w = cu_shapes(n, 1, y, x);
        const auto h = cu_shapes(n, 0, y, x);

        Detection peak = {n, c, float(x), float(y), value, w, h};
        detail::subPixelPeak(cu_logits, y, x, peak);

        peak.x *= 8;
        peak.y *= 8;

        cu_detections[peak_idx] = peak;
    }
};

template <class Fn>
int findPeaks(const Tensor& logits, const float min_confidence, const std::size_t max_peaks, const Fn helper) {
    CHECK(logits.type() == Type::FLOAT32) << "logits must be float32";
    CHECK(logits.device() == Device::CUDA) << "logits must be on CUDA";
    CHECK(logits.shape().order == Ordering::NCHW) << "logits must be NCHW";

    int* num_peaks_d = nullptr;
    cudaMalloc(&num_peaks_d, sizeof(int));
    cudaMemset(num_peaks_d, 0, sizeof(int));

    auto cu_logits = detail::CudaTensor<const float, Ordering::NCHW>(logits);

    dim3 threadsPerBlock(32, 32, 1);
    dim3 numBlocks((cu_logits.w + threadsPerBlock.x - 1) / threadsPerBlock.x, (cu_logits.h + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (cu_logits.n * cu_logits.c + threadsPerBlock.z - 1) / threadsPerBlock.z);

    detail::findPeaksKernel<Fn><<<numBlocks, threadsPerBlock>>>(cu_logits, min_confidence, num_peaks_d, max_peaks, helper);
    cudaDeviceSynchronize();

    int num_peaks_h = 0;
    cudaMemcpy(&num_peaks_h, num_peaks_d, sizeof(int), cudaMemcpyKind::cudaMemcpyDefault);
    cudaDeviceSynchronize();

    return num_peaks_h;
}

}  // namespace detail

std::ostream& operator<<(std::ostream& stream, const Peak& peak) {
    stream << "{ n: " << peak.n << " c: " << peak.c << " y: " << peak.y << " x: " << peak.x << " }";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const Detection& det) {
    stream << "{ peak: " << Peak(det) << " w: " << det.w << " h: " << det.h << " }";
    return stream;
}

template <std::size_t c>
void preprocessImpl(const Tensor& images, const Normalize<c> norm, Tensor& output) {
    CHECK(images.type() == Type::UINT8) << "images must be uint8";
    CHECK(images.device() == Device::CUDA) << "images must be on CUDA";
    CHECK(images.shape().order == Ordering::NHWC) << "images must be NHWC";
    CHECK_EQ(images.shape().c, c) << "image has the wrong number of channels";

    if (output.device() == Device::Empty) {
        auto output_shape = images.shape();
        output_shape.order = Ordering::NCHW;
        output = Tensor(output_shape, Device::CUDA, Type::FLOAT32);
    }

    const auto cu_image = detail::CudaTensor<const unsigned char, Ordering::NHWC>(images);
    auto cu_output = detail::CudaTensor<float, Ordering::NCHW>(output);

    dim3 threadsPerBlock(32, 32, 1);
    dim3 numBlocks((cu_image.w + threadsPerBlock.x - 1) / threadsPerBlock.x, (cu_image.h + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (cu_image.n + threadsPerBlock.z - 1) / threadsPerBlock.z);

    detail::preprocessKernel<c><<<numBlocks, threadsPerBlock>>>(cu_image, norm, cu_output);
}

void preprocess(const Tensor& images, const Normalize<1> norm, Tensor& output) { preprocessImpl<1>(images, norm, output); }

void preprocess(const Tensor& images, const Normalize<3> norm, Tensor& output) { preprocessImpl<3>(images, norm, output); }

void sigmoid(Tensor& logits) {
    CHECK(logits.type() == Type::FLOAT32) << "logits must be float32";
    CHECK(logits.device() == Device::CUDA) << "logits must be on CUDA";
    CHECK(logits.shape().order == Ordering::NCHW) << "logits must be NCHW";

    const auto& shape = logits.shape();
    auto cu_logits = detail::CudaTensor<float, Ordering::NCHW>(logits);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((shape.w + threadsPerBlock.x - 1) / threadsPerBlock.x, (shape.h + threadsPerBlock.y - 1) / threadsPerBlock.y);

    detail::sigmoidKernel<<<numBlocks, threadsPerBlock>>>(cu_logits);

    cudaDeviceSynchronize();
}

void maskFromProbs(const Tensor& probs, Tensor& image) {
    CHECK(image.type() == Type::UINT8) << "image must be uint8";
    CHECK(image.device() == Device::CUDA) << "image must be on CUDA";
    CHECK(image.shape().order == Ordering::NHWC) << "image must be NHWC";
    CHECK(probs.type() == Type::FLOAT32) << "images must be uint8";
    CHECK(probs.device() == Device::CUDA) << "images must be on CUDA";
    CHECK(probs.shape().order == Ordering::NCHW) << "images must be NHWC";

    auto cu_image = detail::CudaTensor<unsigned char, Ordering::NHWC>(image);
    const auto cu_probs = detail::CudaTensor<const float, Ordering::NCHW>(probs);

    CHECK_EQ(cu_image.c, cu_probs.c);

    dim3 threadsPerBlock(32, 32, 1);
    dim3 numBlocks((cu_probs.w + threadsPerBlock.x - 1) / threadsPerBlock.x, (cu_probs.h + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (cu_probs.n + threadsPerBlock.z - 1) / threadsPerBlock.z);

    detail::maskFromProbsKernel<<<numBlocks, threadsPerBlock>>>(cu_probs, cu_image);
}

void colorByProbs(const Tensor& probs, Tensor& image) {
    CHECK(image.type() == Type::UINT8) << "image must be uint8";
    CHECK(image.device() == Device::CUDA) << "image must be on CUDA";
    CHECK(image.shape().order == Ordering::NHWC) << "image must be NHWC";
    CHECK(probs.type() == Type::FLOAT32) << "images must be uint8";
    CHECK(probs.device() == Device::CUDA) << "images must be on CUDA";
    CHECK(probs.shape().order == Ordering::NCHW) << "images must be NHWC";

    auto cu_image = detail::CudaTensor<unsigned char, Ordering::NHWC>(image);
    const auto cu_probs = detail::CudaTensor<const float, Ordering::NCHW>(probs);

    dim3 threadsPerBlock(32, 32, 1);
    dim3 numBlocks((cu_probs.w + threadsPerBlock.x - 1) / threadsPerBlock.x, (cu_probs.h + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (cu_probs.n + threadsPerBlock.z - 1) / threadsPerBlock.z);

    detail::colorByProbsKernel<<<numBlocks, threadsPerBlock>>>(cu_probs, cu_image);
}

void peakFinding(const Tensor& logits, const float min_confidence, const std::size_t max_peaks, std::vector<Peak>& peaks) {
    Peak* cu_peaks = nullptr;
    cudaMalloc(&cu_peaks, sizeof(Peak) * max_peaks);

    auto cu_logits = detail::CudaTensor<const float, Ordering::NCHW>(logits);

    const auto helper = detail::PeakHelper{cu_logits, cu_peaks};
    const int num_peaks = detail::findPeaks(logits, min_confidence, max_peaks, helper);

    peaks.resize(num_peaks);
    cudaMemcpy(peaks.data(), cu_peaks, sizeof(Peak) * num_peaks, cudaMemcpyKind::cudaMemcpyDefault);

    cudaDeviceSynchronize();
}

void objectsAsPoints(const Tensor& logits, const Tensor& shapes, const float min_confidence, const std::size_t max_detections,
                     std::vector<Detection>& detections) {
    CHECK(shapes.type() == Type::FLOAT32) << "shapes must be float32";
    CHECK(shapes.device() == Device::CUDA) << "shapes must be on CUDA";
    CHECK(shapes.shape().order == Ordering::NCHW) << "shapes must be NCHW";

    Detection* cu_detections = nullptr;
    cudaMalloc(&cu_detections, sizeof(Detection) * max_detections);

    auto cu_logits = detail::CudaTensor<const float, Ordering::NCHW>(logits);
    auto cu_shapes = detail::CudaTensor<const float, Ordering::NCHW>(shapes);
    const auto helper = detail::OAPHelper{cu_logits, cu_shapes, cu_detections};

    const int num_peaks = detail::findPeaks(logits, min_confidence, max_detections, helper);

    detections.resize(num_peaks);
    cudaMemcpy(detections.data(), cu_detections, sizeof(Detection) * num_peaks, cudaMemcpyKind::cudaMemcpyDefault);

    cudaDeviceSynchronize();
}
}  // namespace libinfer