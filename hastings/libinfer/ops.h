#pragma once

#include <array>
#include <cstdint>
#include <iosfwd>
#include <vector>

#include "hastings/libinfer/tensor.h"

namespace libinfer {

struct Peak {
    int n, c = 0;
    float x, y = 0.0f;
    float intensity = 0.0f;
};

struct Detection : public Peak {
    float w, h = 0.0f;
};

template <int num_c>
struct Normalize {
    float mean[num_c];
    float std[num_c];
};

std::ostream& operator<<(std::ostream& stream, const Peak& peak);

std::ostream& operator<<(std::ostream& stream, const Detection& det);

void preprocess(const Tensor& images, const Normalize<1> norm, Tensor& output);

void preprocess(const Tensor& images, const Normalize<3> norm, Tensor& output);

void sigmoid(Tensor& logits);

void maskFromProbs(const Tensor& probs, Tensor& image);

void colorByProbs(const Tensor& probs, Tensor& image);

void peakFinding(const Tensor& logits, const float min_confidence, const std::size_t max_peaks, std::vector<Peak>& peaks);

void objectsAsPoints(const Tensor& logits, const Tensor& shapes, const float min_confidence, const std::size_t max_detections,
                     std::vector<Detection>& detections);

// need imageBatch from detections for pose estimation...

}  // namespace libinfer