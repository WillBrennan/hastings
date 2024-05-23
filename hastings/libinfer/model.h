#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "hastings/libinfer/tensor.h"

namespace libinfer {
class Model {
  public:
    using Index = std::int64_t;
    using Path = std::filesystem::path;

    Model(const Path& onnx_path, const bool use_fp16, const Index explicit_batch_size = -1);
    ~Model();

    void forward(std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);
    Index batchSize() const { return explicit_batch_size_; }

  private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
    Path onnx_path_;
    bool use_fp16_;
    Index explicit_batch_size_;
};

}  // namespace libinfer