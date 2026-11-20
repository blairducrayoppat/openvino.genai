// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"

// Small models for the model identity tests. Inputs and outputs carry tensor names, the way a model read from an IR
// does, because the pipeline binds its inputs and reads its outputs by tensor name (model_runner.hpp
// set_tensor("input_ids"), set_tensor("position_ids"), get_tensor("logits")).
namespace model_identity_test_models {

enum class Wiring {
    // input_ids feeds the first input of the operation, position_ids the second.
    Straight,
    // The same two inputs feed the other ports: same graph shape, the inputs swap roles.
    Swapped,
};

inline std::shared_ptr<ov::op::v0::Parameter> named_parameter(const ov::element::Type& type,
                                                              const ov::PartialShape& shape,
                                                              const std::string& name) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
    parameter->set_friendly_name(name);
    parameter->output(0).set_names({name});
    return parameter;
}

/// logits = input_ids - position_ids (Straight) or position_ids - input_ids (Swapped); both inputs share @p shape.
inline std::shared_ptr<ov::Model> make_subtract_model(Wiring wiring,
                                                      const ov::PartialShape& shape = ov::PartialShape{1, 4},
                                                      bool use_add = false) {
    auto input_ids = named_parameter(ov::element::i64, shape, "input_ids");
    auto position_ids = named_parameter(ov::element::i64, shape, "position_ids");
    const ov::Output<ov::Node> lhs = wiring == Wiring::Straight ? input_ids->output(0) : position_ids->output(0);
    const ov::Output<ov::Node> rhs = wiring == Wiring::Straight ? position_ids->output(0) : input_ids->output(0);
    std::shared_ptr<ov::Node> op;
    if (use_add) {
        op = std::make_shared<ov::op::v1::Add>(lhs, rhs);
    } else {
        op = std::make_shared<ov::op::v1::Subtract>(lhs, rhs);
    }
    op->set_friendly_name("op");
    op->output(0).set_names({"logits"});
    auto result = std::make_shared<ov::op::v0::Result>(op);
    result->set_friendly_name("logits");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_ids, position_ids});
}

/// logits = concat(input_ids [1,4], position_ids [1,3]) along axis 1, in that order (Straight) or the other (Swapped).
/// The two inputs differ in shape, so the graph itself records which one comes first.
inline std::shared_ptr<ov::Model> make_concat_model(Wiring wiring) {
    auto input_ids = named_parameter(ov::element::i64, ov::PartialShape{1, 4}, "input_ids");
    auto position_ids = named_parameter(ov::element::i64, ov::PartialShape{1, 3}, "position_ids");
    ov::OutputVector parts = {input_ids, position_ids};
    if (wiring == Wiring::Swapped) {
        parts = {position_ids, input_ids};
    }
    auto concat = std::make_shared<ov::op::v0::Concat>(parts, 1);
    concat->set_friendly_name("op");
    concat->output(0).set_names({"logits"});
    auto result = std::make_shared<ov::op::v0::Result>(concat);
    result->set_friendly_name("logits");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_ids, position_ids});
}

/// Two outputs over the same inputs: "logits" = a + b and "other" = a - b (Straight), or the names swapped (Swapped).
inline std::shared_ptr<ov::Model> make_two_output_model(Wiring wiring) {
    auto a = named_parameter(ov::element::i64, ov::PartialShape{1, 4}, "input_ids");
    auto b = named_parameter(ov::element::i64, ov::PartialShape{1, 4}, "position_ids");
    auto add = std::make_shared<ov::op::v1::Add>(a, b);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(a, b);
    add->output(0).set_names({wiring == Wiring::Straight ? "logits" : "other"});
    subtract->output(0).set_names({wiring == Wiring::Straight ? "other" : "logits"});
    auto add_result = std::make_shared<ov::op::v0::Result>(add);
    auto subtract_result = std::make_shared<ov::op::v0::Result>(subtract);
    return std::make_shared<ov::Model>(ov::ResultVector{add_result, subtract_result}, ov::ParameterVector{a, b});
}

}  // namespace model_identity_test_models
