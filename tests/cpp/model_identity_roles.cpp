// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "continuous_batching/cache/cache_offload_coordinator.hpp"
#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "continuous_batching/cache/model_identity.hpp"
#include "model_identity_test_models.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::genai;
using namespace model_identity_test_models;

namespace {

std::string identity_of(const std::shared_ptr<ov::Model>& model, const char* label) {
    const std::string identity = compute_model_identity(model);
    std::cout << "RECORD identity " << label << " " << identity << std::endl;
    return identity;
}

std::vector<int64_t> run_subtract_model(const std::shared_ptr<ov::Model>& model) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(model, "CPU").create_infer_request();
    ov::Tensor input_ids(ov::element::i64, {1, 4});
    ov::Tensor position_ids(ov::element::i64, {1, 4});
    const int64_t ids[] = {10, 20, 30, 40};
    const int64_t positions[] = {1, 2, 3, 4};
    std::copy(std::begin(ids), std::end(ids), input_ids.data<int64_t>());
    std::copy(std::begin(positions), std::end(positions), position_ids.data<int64_t>());
    // By tensor name, as model_runner.hpp binds them.
    request.set_tensor("input_ids", input_ids);
    request.set_tensor("position_ids", position_ids);
    request.infer();
    const ov::Tensor logits = request.get_tensor("logits");
    return std::vector<int64_t>(logits.data<int64_t>(), logits.data<int64_t>() + logits.get_size());
}

}  // namespace

// Control: the same graph built twice has one identity.
TEST(TestModelIdentityRoles, ControlTheSameModelTwiceGivesTheSameDigest) {
    EXPECT_EQ(identity_of(make_subtract_model(Wiring::Straight), "straight_a"),
              identity_of(make_subtract_model(Wiring::Straight), "straight_b"));
}

// Control: a different operator over the same inputs gives a different identity.
TEST(TestModelIdentityRoles, ControlADifferentOperatorGivesADifferentDigest) {
    EXPECT_NE(identity_of(make_subtract_model(Wiring::Straight), "straight_subtract"),
              identity_of(make_subtract_model(Wiring::Straight, ov::PartialShape{1, 4}, /*use_add=*/true),
                          "straight_add"));
}

// The two models are different functions of their named inputs: same inputs, different logits.
TEST(TestModelIdentityRoles, SwappedInputRolesComputeDifferentLogits) {
    const auto straight = run_subtract_model(make_subtract_model(Wiring::Straight));
    const auto swapped = run_subtract_model(make_subtract_model(Wiring::Swapped));
    EXPECT_EQ(straight, (std::vector<int64_t>{9, 18, 27, 36}));
    EXPECT_EQ(swapped, (std::vector<int64_t>{-9, -18, -27, -36}));
}

// Level 1: both inputs have the same static type and shape, i64 [1,4].
TEST(TestModelIdentityRoles, SwappedInputRolesWithTheSameStaticShapeGiveDifferentDigests) {
    EXPECT_NE(identity_of(make_subtract_model(Wiring::Straight), "static_straight"),
              identity_of(make_subtract_model(Wiring::Swapped), "static_swapped"));
}

// Level 2: both inputs have the same dynamic type and shape, i64 [?,?], as input_ids and position_ids of an LLM do.
TEST(TestModelIdentityRoles, SwappedInputRolesWithTheSameDynamicShapeGiveDifferentDigests) {
    const ov::PartialShape dynamic{ov::Dimension::dynamic(), ov::Dimension::dynamic()};
    EXPECT_NE(identity_of(make_subtract_model(Wiring::Straight, dynamic), "dynamic_straight"),
              identity_of(make_subtract_model(Wiring::Swapped, dynamic), "dynamic_swapped"));
}

// Level 3: the inputs differ in shape ([1,4] and [1,3]), so the graph alone tells the two orders apart.
TEST(TestModelIdentityRoles, SwappedInputsOfDifferentShapesGiveDifferentDigests) {
    EXPECT_NE(identity_of(make_concat_model(Wiring::Straight), "concat_straight"),
              identity_of(make_concat_model(Wiring::Swapped), "concat_swapped"));
}

// Outputs: the same two computations, with the names of the outputs swapped.
TEST(TestModelIdentityRoles, SwappedOutputNamesGiveDifferentDigests) {
    EXPECT_NE(identity_of(make_two_output_model(Wiring::Straight), "outputs_straight"),
              identity_of(make_two_output_model(Wiring::Swapped), "outputs_swapped"));
}

// What a store reads: the namespace (cache_offload_coordinator.hpp group_name) of the two models, same layout.
TEST(TestModelIdentityRoles, SwappedInputRolesOpenDifferentStoreNamespaces) {
    const KVCacheDiskLayout layout({32}, {0});
    const CacheOffloadCoordinator straight(compute_model_identity(make_subtract_model(Wiring::Straight)));
    const CacheOffloadCoordinator swapped(compute_model_identity(make_subtract_model(Wiring::Swapped)));
    const CacheOffloadCoordinator other(
        compute_model_identity(make_subtract_model(Wiring::Straight, ov::PartialShape{1, 4}, /*use_add=*/true)));
    const std::string straight_group = straight.group_name("kv", layout, /*block_tokens=*/4, "CPU");
    std::cout << "RECORD group straight " << straight_group << std::endl;
    std::cout << "RECORD group swapped " << swapped.group_name("kv", layout, 4, "CPU") << std::endl;
    // Control: a different operator opens a different namespace.
    EXPECT_NE(straight_group, other.group_name("kv", layout, 4, "CPU"));
    EXPECT_NE(straight_group, swapped.group_name("kv", layout, 4, "CPU"));
}
