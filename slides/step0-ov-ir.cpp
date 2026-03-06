// Step0: this snippet shows how a simple SDPA(Scaled Dot Product Attention) model is created
// and compiled on OpenVino side.

TEST(MLIRExecution, CompileBasicSDPA) {
    const ov::PartialShape query_shape{2, 4096, 64};
    const ov::PartialShape key_shape{2, 4096, 64};
    const ov::PartialShape value_shape{2, 4096, 64};

    const auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, query_shape);
    const auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, key_shape);
    const auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, value_shape);
    const auto sdpa_mask_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, std::vector<float>{0.0f});
    const auto sdpa_scale_const = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, std::vector<float>{1.0f});
    const auto casual = false;
    const auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query,
                                                                        key,
                                                                        value,
                                                                        sdpa_mask_const,
                                                                        sdpa_scale_const,
                                                                        casual);

    auto model = std::make_shared<ov::Model>(ov::OutputVector{sdpa}, ov::ParameterVector{query, key, value});
    ov::Core core;

    ov::AnyMap device_config;
    device_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
    device_config[ov::enable_profiling.name()] = false;

    auto compiled_model = core.compile_model(model, "GPU", device_config);
}