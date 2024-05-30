#include "hastings/libinfer/model.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <glog/logging.h>

#include <fstream>
#include <iostream>

namespace libinfer {

struct TRTLogger final : nvinfer1::ILogger {
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override final {
        if (severity == Severity::kVERBOSE) {
            return;
        } else if (severity == Severity::kINFO) {
            LOG(INFO) << msg;
        } else if (severity == Severity::kERROR) {
            LOG(ERROR) << msg;
        } else if (severity == Severity::kWARNING) {
            LOG(WARNING) << msg;
        } else if (severity == Severity::kINTERNAL_ERROR) {
            LOG(FATAL) << msg;
        }
    }
};

static TRTLogger logger;

struct Model::Impl {
    using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime>;
    using EnginePtr = std::unique_ptr<nvinfer1::ICudaEngine>;
    using ContextPtr = std::unique_ptr<nvinfer1::IExecutionContext>;

    RuntimePtr runtime;
    EnginePtr engine;
    ContextPtr context;

    Impl(const Path& onnx_path, const bool use_fp16, const std::vector<Tensor>& inputs, const Index explicit_batch_size) {
        using BuilderPtr = std::unique_ptr<nvinfer1::IBuilder>;
        using NetworkPtr = std::unique_ptr<nvinfer1::INetworkDefinition>;
        using ParserPtr = std::unique_ptr<nvonnxparser::IParser>;
        using ProfilePtr = std::unique_ptr<nvinfer1::IOptimizationProfile>;
        using ConfigPtr = std::unique_ptr<nvinfer1::IBuilderConfig>;

        Path engine_path;

        {
            std::string engine_name;
            engine_name = onnx_path.stem().string() + "_";

            for (auto& input : inputs) {
                const auto& shape = input.shape();
                engine_name += "n" + std::to_string(shape.n);
                engine_name += "c" + std::to_string(shape.c);
                engine_name += "h" + std::to_string(shape.h);
                engine_name += "w" + std::to_string(shape.w);
            }

            engine_name += (use_fp16 ? "_fp16" : "");
            engine_name += ".trt.engine";

            engine_path = std::filesystem::current_path() / engine_name;
        }

        if (!std::filesystem::exists(engine_path)) {
            BuilderPtr builder(nvinfer1::createInferBuilder(logger));
            CHECK(builder != nullptr) << "failed to create builder";

            const auto flags =
                explicit_batch_size > 0 ? 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH) : 0U;
            NetworkPtr network(builder->createNetworkV2(flags));
            CHECK(network != nullptr) << "failed to create network";

            LOG(INFO) << "reading onnx file from " << onnx_path;
            ParserPtr parser(nvonnxparser::createParser(*network, logger));
            const auto ret = parser->parseFromFile(onnx_path.c_str(), 0);
            CHECK(ret) << "failed to parse onnx file";

            auto profile = builder->createOptimizationProfile();
            CHECK_EQ(inputs.size(), network->getNbInputs()) << "incorrect number of inputs";

            for (int i = 0; i < inputs.size(); ++i) {
                auto* input = network->getInput(i);
                auto dims = input->getDimensions();

                dims.d[0] = std::max(explicit_batch_size, inputs[i].shape().n);
                dims.d[2] = inputs[i].shape().h;
                dims.d[3] = inputs[i].shape().w;

                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
                profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
            }

            ConfigPtr config(builder->createBuilderConfig());
            config->addOptimizationProfile(profile);

            if (use_fp16) {
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }

            auto serialized_model = builder->buildSerializedNetwork(*network, *config);

            auto out_stream = std::ofstream(engine_path, std::ios::binary);
            CHECK(out_stream) << "failed to create stream to serialize engine";

            LOG(INFO) << "saving trt engine to " << engine_path;

            out_stream.write(reinterpret_cast<const char*>(serialized_model->data()), serialized_model->size());
            out_stream.close();
        }

        LOG(INFO) << "reading trt engine from " << engine_path;
        std::string serialized_model;

        {
            auto in_stream = std::ifstream(engine_path, std::ios::binary);
            CHECK(in_stream) << "failed to read engine";

            serialized_model = std::string((std::istreambuf_iterator<char>(in_stream)), std::istreambuf_iterator<char>());
            in_stream.close();
        }

        {
            runtime = RuntimePtr{nvinfer1::createInferRuntime(logger)};

            auto data_ptr = reinterpret_cast<void*>(serialized_model.data());
            auto num_bytes = serialized_model.size();
            engine = EnginePtr(runtime->deserializeCudaEngine(data_ptr, num_bytes));

            context = ContextPtr(engine->createExecutionContext());
        }
    };
};

Model::Model(const Path& onnx_path, const bool use_fp16, const Index explicit_batch_size)
    : onnx_path_(onnx_path), use_fp16_(use_fp16), explicit_batch_size_(explicit_batch_size) {}

Model::~Model() = default;

void Model::forward(std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    if (impl_ == nullptr) {
        impl_ = std::make_unique<Impl>(onnx_path_, use_fp16_, inputs, explicit_batch_size_);
    };

    auto& engine = *(impl_->engine);
    auto& context = *(impl_->context);

    for (int i = 0; i < inputs.size(); ++i) {
        auto& input = inputs[i];

        CHECK(input.device() == Device::CUDA) << "input must be on cuda";
        CHECK(input.type() == Type::FLOAT32) << "input must be NCHW";
        CHECK(input.shape().order == Ordering::NCHW) << "input must be on cuda";

        const auto name = engine.getIOTensorName(i);
        auto dims = engine.getTensorShape(name);

        dims.d[0] = inputs[i].shape().n;
        dims.d[2] = inputs[i].shape().h;
        dims.d[3] = inputs[i].shape().w;

        context.setTensorAddress(name, reinterpret_cast<void*>(input.data()));
        context.setInputShape(name, dims);
    }

    const auto create_outputs = outputs.empty();

    const auto num_outputs = engine.getNbIOTensors() - inputs.size();
    outputs.reserve(num_outputs);

    for (auto i = 0; i < num_outputs; ++i) {
        const auto name = engine.getIOTensorName(inputs.size() + i);
        auto dims = engine.getTensorShape(name);

        if (create_outputs) {
            Shape shape{Ordering::NCHW, dims.d[0], dims.d[1], std::max(Index(1), dims.d[2]), std::max(Index(1), dims.d[3])};
            outputs.emplace_back(Tensor(shape, Device::CUDA, Type::FLOAT32));
        }

        auto& output = outputs.at(i);
        CHECK_EQ(dims.d[0], output.shape().n) << "invalid output batch-size";
        CHECK_EQ(dims.d[1], output.shape().c) << "invalid output channels";

        CHECK(output.device() == Device::CUDA) << "output must be on cuda";
        CHECK(output.type() == Type::FLOAT32) << "output must be float32";
        CHECK(output.shape().order == Ordering::NCHW) << "output must be NCHW";

        context.setTensorAddress(name, reinterpret_cast<void*>(output.data()));
    }

    // todo(will.brennan): use cuda-stream for performance
    context.enqueueV3(nullptr);
}
}  // namespace libinfer