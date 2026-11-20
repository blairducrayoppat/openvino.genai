// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
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

/// The logits of @p model for input_ids = {10, 20, 30, 40} and position_ids = {1, 2, 3, 4}, both bound by tensor name.
std::vector<int64_t> logits_of(const std::shared_ptr<ov::Model>& model) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(model, "CPU").create_infer_request();
    ov::Tensor input_ids(ov::element::i64, {1, 4});
    ov::Tensor position_ids(ov::element::i64, {1, 4});
    const int64_t ids[] = {10, 20, 30, 40};
    const int64_t positions[] = {1, 2, 3, 4};
    std::copy(std::begin(ids), std::end(ids), input_ids.data<int64_t>());
    std::copy(std::begin(positions), std::end(positions), position_ids.data<int64_t>());
    request.set_tensor("input_ids", input_ids);
    request.set_tensor("position_ids", position_ids);
    request.infer();
    const ov::Tensor logits = request.get_tensor("logits");
    return std::vector<int64_t>(logits.data<int64_t>(), logits.data<int64_t>() + logits.get_size());
}

std::string group_of(const std::shared_ptr<ov::Model>& model, const char* label) {
    const KVCacheDiskLayout layout({32}, {0});
    const std::string group =
        CacheOffloadCoordinator(compute_model_identity(model)).group_name("kv", layout, /*block_tokens=*/4, "CPU");
    std::cout << "RECORD group " << label << " " << group << std::endl;
    return group;
}

const ov::PartialShape DYNAMIC{ov::Dimension::dynamic(), ov::Dimension::dynamic()};

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

// The variants are different models for the pipeline: the same named inputs give different logits.
TEST(TestModelIdentityRoles, TheVariantsComputeDifferentLogitsForTheSameNamedInputs) {
    EXPECT_EQ(logits_of(make_subtract_model(Wiring::Straight)), (std::vector<int64_t>{9, 18, 27, 36}));
    EXPECT_EQ(logits_of(make_subtract_model(Wiring::Rewired)), (std::vector<int64_t>{-9, -18, -27, -36}));
    EXPECT_EQ(logits_of(make_subtract_model(Wiring::NamesSwapped)), (std::vector<int64_t>{-9, -18, -27, -36}));
    EXPECT_EQ(logits_of(make_two_output_model(Wiring::Straight)), (std::vector<int64_t>{11, 22, 33, 44}));
    EXPECT_EQ(logits_of(make_two_output_model(Wiring::NamesSwapped)), (std::vector<int64_t>{9, 18, 27, 36}));
}

// Rewired, level 1: both inputs i64 [1,4].
TEST(TestModelIdentityRoles, RewiredInputsWithTheSameStaticShapeGiveDifferentDigests) {
    EXPECT_NE(identity_of(make_subtract_model(Wiring::Straight), "static_straight"),
              identity_of(make_subtract_model(Wiring::Rewired), "static_rewired"));
}

// Rewired, level 2: both inputs i64 [?,?], as input_ids and position_ids of an LLM are.
TEST(TestModelIdentityRoles, RewiredInputsWithTheSameDynamicShapeGiveDifferentDigests) {
    EXPECT_NE(identity_of(make_subtract_model(Wiring::Straight, DYNAMIC), "dynamic_straight"),
              identity_of(make_subtract_model(Wiring::Rewired, DYNAMIC), "dynamic_rewired"));
}

// Rewired, level 3: inputs of different shapes ([1,4] and [1,3]).
TEST(TestModelIdentityRoles, RewiredInputsOfDifferentShapesGiveDifferentDigests) {
    EXPECT_NE(identity_of(make_concat_model(Wiring::Straight), "concat_straight"),
              identity_of(make_concat_model(Wiring::Rewired), "concat_rewired"));
}

// Names swapped, level 1: both inputs i64 [1,4].
TEST(TestModelIdentityRoles, SwappedInputNamesWithTheSameStaticShapeGiveDifferentDigests) {
    EXPECT_NE(identity_of(make_subtract_model(Wiring::Straight), "static_straight"),
              identity_of(make_subtract_model(Wiring::NamesSwapped), "static_names_swapped"));
}

// Names swapped, level 2: both inputs i64 [?,?].
TEST(TestModelIdentityRoles, SwappedInputNamesWithTheSameDynamicShapeGiveDifferentDigests) {
    EXPECT_NE(identity_of(make_subtract_model(Wiring::Straight, DYNAMIC), "dynamic_straight"),
              identity_of(make_subtract_model(Wiring::NamesSwapped, DYNAMIC), "dynamic_names_swapped"));
}

// Names swapped, level 3: inputs of different shapes ([1,4] and [1,3]).
TEST(TestModelIdentityRoles, SwappedInputNamesOfDifferentShapesGiveDifferentDigests) {
    EXPECT_NE(identity_of(make_concat_model(Wiring::Straight), "concat_straight"),
              identity_of(make_concat_model(Wiring::NamesSwapped), "concat_names_swapped"));
}

// Outputs: the same two computations with the two output names exchanged.
TEST(TestModelIdentityRoles, SwappedOutputNamesGiveDifferentDigests) {
    EXPECT_NE(identity_of(make_two_output_model(Wiring::Straight), "outputs_straight"),
              identity_of(make_two_output_model(Wiring::NamesSwapped), "outputs_names_swapped"));
}

// What a store reads: the namespace (cache_offload_coordinator.hpp group_name), same layout, block size and device.
TEST(TestModelIdentityRoles, EachVariantOpensItsOwnStoreNamespace) {
    const std::string straight = group_of(make_subtract_model(Wiring::Straight), "straight");
    EXPECT_EQ(straight, group_of(make_subtract_model(Wiring::Straight), "straight_again")) << "control";
    EXPECT_NE(straight, group_of(make_subtract_model(Wiring::Straight, ov::PartialShape{1, 4}, true), "add"))
        << "control";
    EXPECT_NE(straight, group_of(make_subtract_model(Wiring::Rewired), "rewired"));
    EXPECT_NE(straight, group_of(make_subtract_model(Wiring::NamesSwapped), "names_swapped"));
    EXPECT_NE(group_of(make_two_output_model(Wiring::Straight), "outputs_straight"),
              group_of(make_two_output_model(Wiring::NamesSwapped), "outputs_names_swapped"));
}
