// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <utility>

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
    // The first parameter is input_ids and feeds the first port, the second is position_ids and feeds the second.
    Straight,
    // The same parameters feed the other ports: the wiring differs, the names stay with their parameters.
    Rewired,
    // The same parameters feed the same ports, but the two names are exchanged: node for node and edge for edge the
    // graph is the Straight one, yet input_ids and position_ids swap roles.
    NamesSwapped,
};

inline std::shared_ptr<ov::op::v0::Parameter> named_parameter(const ov::element::Type& type,
                                                              const ov::PartialShape& shape,
                                                              const std::string& name) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
    parameter->set_friendly_name(name);
    parameter->output(0).set_names({name});
    return parameter;
}

/// Names of the first and second parameter.
inline std::pair<std::string, std::string> parameter_names(Wiring wiring) {
    if (wiring == Wiring::NamesSwapped) {
        return {"position_ids", "input_ids"};
    }
    return {"input_ids", "position_ids"};
}

/// logits = first - second (Straight, NamesSwapped) or second - first (Rewired); both inputs share @p shape.
/// Straight computes input_ids - position_ids; Rewired and NamesSwapped compute position_ids - input_ids.
inline std::shared_ptr<ov::Model> make_subtract_model(Wiring wiring,
                                                      const ov::PartialShape& shape = ov::PartialShape{1, 4},
                                                      bool use_add = false) {
    const auto names = parameter_names(wiring);
    auto first = named_parameter(ov::element::i64, shape, names.first);
    auto second = named_parameter(ov::element::i64, shape, names.second);
    const ov::Output<ov::Node> lhs = wiring == Wiring::Rewired ? second->output(0) : first->output(0);
    const ov::Output<ov::Node> rhs = wiring == Wiring::Rewired ? first->output(0) : second->output(0);
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
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{first, second});
}

/// logits = concat(first [1,4], second [1,3]) along axis 1 (Straight, NamesSwapped) or concat(second, first)
/// (Rewired). The two inputs differ in shape.
inline std::shared_ptr<ov::Model> make_concat_model(Wiring wiring) {
    const auto names = parameter_names(wiring);
    auto first = named_parameter(ov::element::i64, ov::PartialShape{1, 4}, names.first);
    auto second = named_parameter(ov::element::i64, ov::PartialShape{1, 3}, names.second);
    ov::OutputVector parts = {first, second};
    if (wiring == Wiring::Rewired) {
        parts = {second, first};
    }
    auto concat = std::make_shared<ov::op::v0::Concat>(parts, 1);
    concat->set_friendly_name("op");
    concat->output(0).set_names({"logits"});
    auto result = std::make_shared<ov::op::v0::Result>(concat);
    result->set_friendly_name("logits");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{first, second});
}

/// Two outputs over the same inputs: "logits" = a + b and "other" = a - b (Straight), or the two output names
/// exchanged (NamesSwapped).
inline std::shared_ptr<ov::Model> make_two_output_model(Wiring wiring) {
    auto a = named_parameter(ov::element::i64, ov::PartialShape{1, 4}, "input_ids");
    auto b = named_parameter(ov::element::i64, ov::PartialShape{1, 4}, "position_ids");
    auto add = std::make_shared<ov::op::v1::Add>(a, b);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(a, b);
    add->output(0).set_names({wiring == Wiring::NamesSwapped ? "other" : "logits"});
    subtract->output(0).set_names({wiring == Wiring::NamesSwapped ? "logits" : "other"});
    auto add_result = std::make_shared<ov::op::v0::Result>(add);
    auto subtract_result = std::make_shared<ov::op::v0::Result>(subtract);
    return std::make_shared<ov::Model>(ov::ResultVector{add_result, subtract_result}, ov::ParameterVector{a, b});
}

}  // namespace model_identity_test_models
