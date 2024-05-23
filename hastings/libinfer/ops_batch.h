#pragma once

#include <vector>

#include "hastings/libinfer/tensor.h"

namespace libinfer {

struct Crop {
    float tl[2], shape[2];
};

void cropFromImage(const Tensor& image_bgr, const std::vector<Crop>& crops, const int maxCrops, const int cropWidth,
                   Tensor& crops_image);
}  // namespace libinfer