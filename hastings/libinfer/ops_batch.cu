#include <glog/logging.h>

#include "hastings/libinfer/cu_tensor.h"
#include "hastings/libinfer/ops_batch.h"

namespace libinfer {
namespace detail {
__global__ void cropKernel(CudaTensor<const unsigned char, Ordering::NHWC> cu_image_bgr, Crop* cu_crops,
                           const int num_crops, CudaTensor<unsigned char, Ordering::NHWC> cu_image_crops) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc = blockIdx.z * blockDim.z + threadIdx.z;
    const int n = nc / cu_image_crops.c;
    const int c = nc % cu_image_crops.c;

    const int crop_w = cu_image_crops.w;
    const int crop_h = cu_image_crops.h;

    if (x >= crop_w || y >= crop_h || c >= cu_image_crops.c || n >= num_crops) {
        return;
    }

    const auto& crop = cu_crops[n];

    const float bgr_x = crop.tl[0] + (x * crop.shape[0]) / crop_w;
    const float bgr_y = crop.tl[1] + (y * crop.shape[1]) / crop_h;

    const bool inImageX = bgr_x >= 0.0 && bgr_x < cu_image_bgr.w;
    const bool inImageY = bgr_y >= 0.0 && bgr_y < cu_image_bgr.h;
    const bool inImage = inImageX && inImageY;

    if (!inImage) {
        cu_image_crops(n, c, y, x) = 127;
        return;
    }

    const int x0 = int(bgr_x);
    const int y0 = int(bgr_y);
    const int x1 = min(x0 + 1, cu_image_bgr.w - 1);
    const int y1 = min(y0 + 1, cu_image_bgr.h - 1);

    const float v00 = cu_image_bgr(0, c, y0, x0);
    const float v01 = cu_image_bgr(0, c, y0, x1);
    const float v10 = cu_image_bgr(0, c, y1, x0);
    const float v11 = cu_image_bgr(0, c, y1, x1);

    const float interp0 = v00 + (v01 - v00) * (bgr_x - x0);
    const float interp1 = v10 + (v11 - v10) * (bgr_x - x0);
    cu_image_crops(n, c, y, x) = interp0 + (interp1 - interp0) * (bgr_y - y0);
}
}  // namespace detail

void cropFromImage(const Tensor& image_bgr, const std::vector<Crop>& crops, const int maxCrops, const int cropWidth,
                   Tensor& crops_image) {
    CHECK(image_bgr.type() == Type::UINT8) << "images must be uint8";
    CHECK(image_bgr.device() == Device::CUDA) << "images must be on CUDA";
    CHECK(image_bgr.shape().order == Ordering::NHWC) << "images must be NHWC";

    const auto crops_shape = Shape{Ordering::NHWC, maxCrops, 3, cropWidth, cropWidth};
    if (crops_image.shape() != crops_shape) {
        crops_image = Tensor(crops_shape, Device::CUDA, Type::UINT8);
    }

    const auto cu_image_bgr = detail::CudaTensor<const unsigned char, Ordering::NHWC>(image_bgr);
    auto cu_image_crops = detail::CudaTensor<unsigned char, Ordering::NHWC>(crops_image);

    Crop* cu_crops = nullptr;
    const auto numCrops = std::min(maxCrops, int(crops.size()));
    const auto numBytes = sizeof(Crop) * numCrops;

    cudaMalloc(&cu_crops, numBytes);
    cudaMemcpy(cu_crops, crops.data(), numBytes, cudaMemcpyDefault);

    cudaDeviceSynchronize();

    dim3 threadsPerBlock(32, 32, 1);
    dim3 numBlocks((cu_image_crops.w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (cu_image_crops.h + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (cu_image_crops.c * cu_image_crops.n + threadsPerBlock.z - 1) / threadsPerBlock.z);

    detail::cropKernel<<<numBlocks, threadsPerBlock>>>(cu_image_bgr, cu_crops, numCrops, cu_image_crops);
    cudaDeviceSynchronize();  // Synchronize to catch errors
}
}  // namespace libinfer