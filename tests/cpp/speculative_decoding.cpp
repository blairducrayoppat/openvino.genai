// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include "gtest/gtest.h"

#include "openvino/genai/speculative_decoding/perf_metrics.hpp"
#include "speculative_decoding/continuous_batching/mtp_strategy.hpp"
#include "speculative_decoding/continuous_batching/pipeline_impl.hpp"
#include "scheduler_test_utils.hpp"
#include "utils.hpp"

namespace {

// Sits where the offload store sits: it is handed the blocks a forward pass has completed.
struct CompletedBlocksRecorder : ov::genai::IOverwrittenBlockObserver {
    std::vector<size_t> hashes;

    void on_blocks_overwritten(size_t, const ov::genai::BlocksPerLayer&) override {}

    void on_blocks_published(size_t hash, const ov::genai::BlocksPerLayer&) override {
        hashes.push_back(hash);
    }
};

}  // namespace

class CBForSDTest : public testing::Test, public ov::genai::ContinuousBatchingPipeline {
protected:
    class PipelineTestInstance : public ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl {
    public:
        PipelineTestInstance() {
            m_sampler = std::make_shared<ov::genai::Sampler>();
        };

        ov::genai::GenerationHandle add_request(uint64_t request_id, const ov::Tensor& input_ids) {
            auto sampling_params = ov::genai::utils::get_greedy_config();
            sampling_params.num_assistant_tokens = 1;

            ov::genai::SequenceGroup::Ptr sequence_group = std::make_shared<ov::genai::SequenceGroup>(request_id, input_ids,
                                                                                sampling_params);

            {
                std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
                m_awaiting_requests.push_back(sequence_group);
            }
            pull_awaiting_requests();
            return std::make_shared<ov::genai::GenerationHandleImpl>(sequence_group->get_generation_stream(), sampling_params);
        };

        void register_generated_tokens(uint64_t request_id, const std::vector<int64_t>& token_ids) {
            auto& logit_processor = m_sampler->get_logit_processor(request_id);
            for (auto token_id : token_ids) {
                logit_processor.register_new_generated_token(token_id);
            }
        }

        void enable_mtp_mode() {
            mtp_mode_enabled = true;
        }

        bool is_waiting(uint64_t request_id) const {
            auto request_it = std::find_if(m_requests.begin(), m_requests.end(), [request_id](const ov::genai::SequenceGroup::Ptr& request) {
                return request->get_request_id() == request_id;
            });
            OPENVINO_ASSERT(request_it != m_requests.end(), "Request is not found");
            return (*request_it)->is_waiting();
        }

        bool can_publish_kv_only_completed_blocks() const {
            return _can_publish_kv_only_completed_blocks();
        }

        void set_scheduler(std::shared_ptr<ov::genai::CacheOrchestrator> cache_orchestrator,
                           const ov::genai::SchedulerConfig& scheduler_config) {
            m_scheduler = std::make_shared<ov::genai::ContinuousBatchingScheduler>(cache_orchestrator, scheduler_config);
        }

        // Schedules a new request (the prompt and, if given, draft candidates to validate), stands in for its forward
        // pass, optionally removes the last generated tokens the way a stop string does, and runs the publication step
        // of the pipeline. Returns the prefix hashes of the blocks an observer of the KV block manager was handed.
        std::vector<size_t> hashes_handed_over_after_one_step(const ov::genai::SchedulerConfig& scheduler_config,
                                                              bool is_validation_mode_enabled,
                                                              std::vector<int64_t> prompt,
                                                              const std::vector<int64_t>& candidates = {},
                                                              size_t tokens_trimmed_after_forward = 0) {
            m_is_validation_mode_enabled = is_validation_mode_enabled;
            auto cache_orchestrator = init_cache_orchestrator(scheduler_config);
            set_scheduler(cache_orchestrator, scheduler_config);
            CompletedBlocksRecorder recorder;
            auto& block_manager = cache_orchestrator->get_block_manager(ov::genai::CacheType::KV_CACHE);
            block_manager.set_overwritten_block_observer(&recorder);

            auto sampling_params = ov::genai::utils::get_greedy_config();
            sampling_params.num_assistant_tokens = 1;
            m_requests.push_back(std::make_shared<ov::genai::SequenceGroup>(
                /*request_id=*/0, ov::Tensor(ov::element::i64, {prompt.size()}, prompt.data()), sampling_params));
            const auto sequence = m_requests.front()->get_sequences().front();
            for (const int64_t candidate : candidates) {
                sequence->append_token(candidate, 0.0f);
            }
            m_requests.front()->set_num_validated_tokens(candidates.size());
            const auto scheduler_output = m_scheduler->schedule(m_requests);
            OPENVINO_ASSERT(m_requests.front()->get_num_scheduled_tokens() == prompt.size() + candidates.size(),
                            "The test expects the prompt and every candidate in one step");
            m_requests.front()->finish_iteration();
            sequence->remove_last_tokens(static_cast<int>(tokens_trimmed_after_forward));
            _publish_completed_cache_blocks(scheduler_output);

            block_manager.set_overwritten_block_observer(nullptr);
            m_scheduler->free_sequence(sequence->get_id());
            m_requests.clear();
            return recorder.hashes;
        }
    };

    class MtpPipelineTestInstance : public ContinuousBatchingPipeline::MtpDecodingImpl {
    public:
        MtpPipelineTestInstance() = default;
        using MtpDecodingImpl::make_draft_generation_config;
        using MtpDecodingImpl::make_draft_sampling_params;
    };

    PipelineTestInstance m_pipeline = PipelineTestInstance();
};

TEST(SDPerModelsPerfMetrics, DraftOverheadDiagnostics) {
    ov::genai::SDPerModelsPerfMetrics metrics;
    metrics.num_draft_tokens = 5;
    metrics.num_accepted_tokens = 3;

    metrics.main_model_metrics.raw_metrics.m_durations = {ov::genai::MicroSeconds(1000.0f)};
    metrics.main_model_metrics.raw_metrics.m_batch_sizes = {4};
    metrics.main_model_metrics.raw_metrics.m_inference_durations = {ov::genai::MicroSeconds(4000.0f)};

    metrics.draft_model_metrics.raw_metrics.m_durations = {ov::genai::MicroSeconds(1000.0f),
                                                            ov::genai::MicroSeconds(1000.0f)};
    metrics.draft_model_metrics.raw_metrics.m_batch_sizes = {4, 4};
    metrics.draft_model_metrics.raw_metrics.m_inference_durations = {ov::genai::MicroSeconds(2000.0f)};

    EXPECT_EQ(metrics.get_num_draft_processed_tokens(), 8);
    EXPECT_FLOAT_EQ(metrics.get_draft_processed_to_candidate_ratio(), 8.0f / 5.0f);
    EXPECT_FLOAT_EQ(metrics.get_draft_to_main_inference_duration_ratio(), 0.5f);
}

TEST(SDPerModelsPerfMetrics, DraftOverheadDiagnosticsReturnNanWithoutDenominator) {
    ov::genai::SDPerModelsPerfMetrics metrics;

    EXPECT_TRUE(std::isnan(metrics.get_draft_processed_to_candidate_ratio()));
    EXPECT_TRUE(std::isnan(metrics.get_draft_to_main_inference_duration_ratio()));
}

TEST_F(CBForSDTest, DraftPipelineKeepsKvOnlyCompletedBlocksUnpublished) {
    EXPECT_FALSE(m_pipeline.can_publish_kv_only_completed_blocks());
}

// The publication step of a fast-draft pipeline is off by default. It has to run for an offload store that takes
// completed blocks, because it is what hands a block over once the forward pass has covered it; without that neither
// the main model's pipeline (validation mode) nor the draft model's stores anything.
TEST_F(CBForSDTest, FastDraftPipelineHandsCompletedBlocksToAStoreThatTakesThem) {
    const std::vector<int64_t> prompt = {0, 1, 2, 3, 4, 5, 6, 7, 8};  // two full blocks and one token of a third
    for (const bool is_validation_mode_enabled : {true, false}) {
        SCOPED_TRACE(is_validation_mode_enabled ? "main model pipeline" : "draft model pipeline");
        ov::genai::SchedulerConfig scheduler_config;
        scheduler_config.num_kv_blocks = 8;
        scheduler_config.enable_prefix_caching = true;
        EXPECT_TRUE(m_pipeline.hashes_handed_over_after_one_step(scheduler_config, is_validation_mode_enabled, prompt).empty())
            << "no offload store";

        scheduler_config.use_cache_offload = true;
        EXPECT_TRUE(m_pipeline.hashes_handed_over_after_one_step(scheduler_config, is_validation_mode_enabled, prompt).empty())
            << "a store without store_completed_blocks takes blocks only when they are overwritten";

        scheduler_config.cache_offload_config.store_completed_blocks = true;
        EXPECT_EQ(m_pipeline.hashes_handed_over_after_one_step(scheduler_config, is_validation_mode_enabled, prompt).size(), 2u)
            << "both blocks the prompt completed must be handed to the store";
    }
}

// A stop string removes generated tokens after the forward pass that processed them, so the processed count the
// publication step sees is past the tokens the sequence still holds. The step must hand over the blocks those
// tokens still fill and leave the rest alone.
TEST_F(CBForSDTest, FastDraftPipelineHandsOverOnlyBlocksTheSequenceStillHoldsAfterAStopStringTrim) {
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.num_kv_blocks = 8;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.use_cache_offload = true;
    scheduler_config.cache_offload_config.store_completed_blocks = true;
    std::vector<size_t> hashes;
    EXPECT_NO_THROW(hashes = m_pipeline.hashes_handed_over_after_one_step(scheduler_config,
                                                                          /*is_validation_mode_enabled=*/true,
                                                                          /*prompt=*/{0, 1, 2, 3},
                                                                          /*candidates=*/{4, 5, 6, 7},
                                                                          /*tokens_trimmed_after_forward=*/4));
    EXPECT_EQ(hashes.size(), 1u) << "only the prompt block is still described by the tokens of the sequence";
}

// A pipeline with linear-attention state takes the other branch of the publication step; the switch for KV-only
// sequences stays off for it whatever the store settings are.
TEST_F(CBForSDTest, FastDraftPipelineWithLinearAttentionStateKeepsKvOnlyPublicationOff) {
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.num_kv_blocks = 8;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.use_cache_offload = true;
    scheduler_config.cache_offload_config.store_completed_blocks = true;
    m_pipeline.set_scheduler(init_hybrid_cache_orchestrator(scheduler_config), scheduler_config);
    EXPECT_FALSE(m_pipeline.can_publish_kv_only_completed_blocks());
    m_pipeline.set_scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);
    EXPECT_TRUE(m_pipeline.can_publish_kv_only_completed_blocks());
}

TEST_F(CBForSDTest, MtpDraftConfigWithoutTokenLimitIgnoresEos) {
    ov::genai::GenerationConfig config;
    config.num_assistant_tokens = 2;
    config.stop_strings = {"stop"};

    const auto draft_config = MtpPipelineTestInstance::make_draft_generation_config(config);

    EXPECT_EQ(config.max_new_tokens, SIZE_MAX);
    EXPECT_EQ(config.max_length, SIZE_MAX);
    EXPECT_EQ(draft_config.max_new_tokens, SIZE_MAX - 1);
    EXPECT_EQ(draft_config.max_length, SIZE_MAX);
    EXPECT_TRUE(draft_config.ignore_eos);
    EXPECT_TRUE(draft_config.stop_strings.empty());
    EXPECT_NO_THROW(draft_config.validate());
}

TEST_F(CBForSDTest, MtpDraftConfigPreservesExplicitTokenLimit) {
    ov::genai::GenerationConfig config;
    config.num_assistant_tokens = 2;
    config.max_new_tokens = 200;
    const auto draft_config = MtpPipelineTestInstance::make_draft_generation_config(config);

    EXPECT_EQ(draft_config.max_new_tokens, 200);
    EXPECT_NO_THROW(draft_config.validate());

    config.max_new_tokens = SIZE_MAX;
    config.max_length = 512;
    const auto draft_length_config = MtpPipelineTestInstance::make_draft_generation_config(config);

    EXPECT_EQ(draft_length_config.max_new_tokens, SIZE_MAX);
    EXPECT_EQ(draft_length_config.max_length, 512);
    EXPECT_NO_THROW(draft_length_config.validate());
}

// The draft copy of a request samples from a stream independent of the main request's, whose draws accept or reject
// the drafted tokens. FastDraft and EAGLE3 use make_draft_sampling_params, MTP make_draft_generation_config.
TEST_F(CBForSDTest, DraftConfigsSampleFromAnIndependentStream) {
    ov::genai::GenerationConfig config;
    config.rng_seed = 42;
    const size_t draft_seed = ov::genai::detail::proposal_rng_seed(config.rng_seed);

    EXPECT_NE(draft_seed, config.rng_seed);
    EXPECT_EQ(MtpPipelineTestInstance::make_draft_sampling_params(config).rng_seed, draft_seed);
    EXPECT_EQ(MtpPipelineTestInstance::make_draft_generation_config(config).rng_seed, draft_seed);
}

// A sampled draft records its proposals without log-probs; a greedy draft keeps them for
// assistant_confidence_threshold.
TEST_F(CBForSDTest, OnlyGreedyDraftConfigsKeepLogProbs) {
    ov::genai::GenerationConfig config;
    config.logprobs = 3;
    EXPECT_EQ(MtpPipelineTestInstance::make_draft_sampling_params(config).logprobs, 3u);
    config.do_sample = true;
    EXPECT_EQ(MtpPipelineTestInstance::make_draft_sampling_params(config).logprobs, 0u);
    EXPECT_EQ(MtpPipelineTestInstance::make_draft_generation_config(config).logprobs, 0u);
}

TEST(MtpDraftUpdatePlan, PreservesAcceptedPrefixAfterPartialRejection) {
    struct TestCase {
        size_t removed_draft_tokens;
        size_t accepted_draft_tokens;
        size_t hidden_state_start;
        size_t processed_tokens_to_rewind;
    };
    constexpr size_t hidden_state_len = 5;
    constexpr size_t num_draft_tokens = hidden_state_len - 1;
    constexpr size_t processed_tokens_before_update = 100;
    const std::vector<TestCase> test_cases{
        {4, 0, 0, 3},  // first candidate rejected
        {2, 2, 2, 1},  // two candidates accepted
        {1, 3, 3, 0},  // only the unforwarded tail candidate rejected
    };

    for (const auto& test_case : test_cases) {
        SCOPED_TRACE(test_case.removed_draft_tokens);
        const auto plan =
            ov::genai::detail::make_mtp_draft_update_plan(hidden_state_len, test_case.removed_draft_tokens);

        EXPECT_EQ(plan.hidden_state_start, test_case.hidden_state_start);
        EXPECT_EQ(plan.hidden_state_count, 1);
        EXPECT_EQ(plan.processed_tokens_to_rewind, test_case.processed_tokens_to_rewind);
        EXPECT_EQ(plan.num_tokens_to_validate, 0);
        EXPECT_EQ(plan.hidden_state_count, plan.num_tokens_to_validate + 1);
        EXPECT_EQ(processed_tokens_before_update - plan.processed_tokens_to_rewind,
                  processed_tokens_before_update - test_case.removed_draft_tokens + 1);
        EXPECT_EQ(test_case.accepted_draft_tokens,
                  num_draft_tokens - test_case.removed_draft_tokens);
        if (test_case.accepted_draft_tokens > 0) {
            EXPECT_LT(plan.hidden_state_count, test_case.accepted_draft_tokens + 1);
        }
    }
}

TEST(MtpDraftUpdatePlan, FullAcceptanceProcessesOnlyUnforwardedTailAndBonus) {
    constexpr size_t hidden_state_len = 5;
    constexpr size_t num_draft_tokens = hidden_state_len - 1;
    constexpr size_t processed_tokens_before_update = 100;
    const auto plan = ov::genai::detail::make_mtp_draft_update_plan(hidden_state_len, 0);

    EXPECT_EQ(plan.hidden_state_start, num_draft_tokens - 1);
    EXPECT_EQ(plan.hidden_state_count, 2);
    EXPECT_EQ(plan.processed_tokens_to_rewind, 0);
    EXPECT_EQ(plan.num_tokens_to_validate, 1);
    EXPECT_EQ(plan.hidden_state_count, plan.num_tokens_to_validate + 1);
    EXPECT_LT(plan.hidden_state_count, hidden_state_len);
    EXPECT_EQ(processed_tokens_before_update - plan.processed_tokens_to_rewind,
              processed_tokens_before_update);
}

namespace {
template <typename Pipeline>
void expect_mtp_request_rejected(Pipeline& pipeline,
                                 const ov::genai::GenerationConfig& config,
                                 const std::string& expected_message) {
    // The default-constructed pipeline and empty tensor are deliberate: unsupported configurations
    // must be rejected before add_request touches model state or validates its input tensor.
    try {
        pipeline.add_request(0, ov::Tensor{}, config);
        FAIL() << "Expected MTP request to be rejected at admission";
    } catch (const ov::Exception& exception) {
        EXPECT_NE(std::string(exception.what()).find(expected_message), std::string::npos)
            << exception.what();
    }
}

ov::genai::GenerationConfig valid_mtp_config() {
    auto config = ov::genai::utils::get_greedy_config();
    config.num_assistant_tokens = 1;
    return config;
}
}  // namespace

TEST_F(CBForSDTest, MtpAdmissionRejectsConfidenceThreshold) {
    MtpPipelineTestInstance pipeline;
    auto config = valid_mtp_config();
    config.assistant_confidence_threshold = 0.5f;

    expect_mtp_request_rejected(pipeline, config, "assistant_confidence_threshold must be 0.f");
}

TEST_F(CBForSDTest, MtpAdmissionRejectsTreeSearch) {
    MtpPipelineTestInstance pipeline;
    auto config = valid_mtp_config();
    config.tree_depth = 2;

    expect_mtp_request_rejected(pipeline, config, "does not support tree search");
}

TEST_F(CBForSDTest, MtpAdmissionRejectsBeamSearch) {
    MtpPipelineTestInstance pipeline;
    auto config = valid_mtp_config();
    config.num_beams = 2;

    expect_mtp_request_rejected(pipeline, config, "supports greedy and multinomial decoding only");
}

TEST_F(CBForSDTest, MtpAdmissionRejectsParallelSampling) {
    MtpPipelineTestInstance pipeline;
    auto config = valid_mtp_config();
    config.num_return_sequences = 2;

    expect_mtp_request_rejected(pipeline, config, "num_return_sequences must be 1");
}

TEST_F(CBForSDTest, MtpAdmissionRejectsZeroAssistantTokens) {
    MtpPipelineTestInstance pipeline;
    auto config = valid_mtp_config();
    config.num_assistant_tokens = 0;

    expect_mtp_request_rejected(pipeline, config, "num_assistant_tokens > 0");
}

TEST_F(CBForSDTest, MtpSupportedConfigPassesAdmissionValidation) {
    MtpPipelineTestInstance pipeline;
    ov::Tensor invalid_input(ov::element::f32, ov::Shape{});
    auto sampled_config = valid_mtp_config();
    sampled_config.do_sample = true;
    sampled_config.top_k = 20;

    for (const auto& config : {valid_mtp_config(), sampled_config}) {
        try {
            pipeline.add_request(0, invalid_input, config);
            FAIL() << "Expected the scalar test tensor to be rejected after admission validation";
        } catch (const ov::Exception& exception) {
            EXPECT_NE(std::string(exception.what()).find("MTP draft input embeds expect shape"), std::string::npos)
                << exception.what();
        }
    }
}

TEST_F(CBForSDTest, init_sequence_by_not_empty__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto before = m_pipeline.get_generated_requests();
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

TEST_F(CBForSDTest, init_sequence_by_empty__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = {};
    std::vector<float> log_probs = {};
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto before = m_pipeline.get_generated_requests();
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_EQ(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_EQ(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

TEST_F(CBForSDTest, no_updated_tokens__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    ov::genai::GeneratedSequences candidate_1{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_EQ(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_EQ(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

TEST_F(CBForSDTest, remove_tokens__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    tokens = { 0, 1 };
    log_probs = { 0.1f, 0.2f };
    ov::genai::GeneratedSequences candidate_1{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 1);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

TEST_F(CBForSDTest, mtp_rejection_pauses_draft_generation) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = {0, 1, 2};
    std::vector<float> log_probs = {0.1f, 0.2f, 0.3f};
    ov::genai::GeneratedSequences candidate{{0, ov::genai::GeneratedSequence(tokens, log_probs)}};

    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);
    ASSERT_FALSE(m_pipeline.is_waiting(0));

    m_pipeline.enable_mtp_mode();
    tokens = {0, 1};
    log_probs = {0.1f, 0.2f};
    ov::genai::GeneratedSequences rejected_candidate{{0, ov::genai::GeneratedSequence(tokens, log_probs)}};

    update_result = m_pipeline.update_request(0, rejected_candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 1);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);
    ASSERT_TRUE(m_pipeline.is_waiting(0));
}

TEST_F(CBForSDTest, remove_and_replace_tokens__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    tokens = { 0, 1, 4 };
    log_probs = { 0.1f, 0.2f, 0.4f };
    ov::genai::GeneratedSequences candidate_1{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 1);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 1);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

TEST_F(CBForSDTest, add_tokens__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    tokens = { 0, 1, 2, 3, 4 };
    log_probs = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
    ov::genai::GeneratedSequences candidate_1{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 2);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
}

// The main sampler verifies each candidate against the q(.) stored at its position, so proposals must stay attached
// to the tokens they were drafted for across update rounds.
TEST_F(CBForSDTest, draft_proposals_follow_inserted_tokens__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    ASSERT_EQ(m_pipeline.update_request(0, candidate, false).inserted_tokens_cnt, 3);

    // The draft proposed tokens 3 and 4 this round.
    tokens = { 0, 1, 2, 3, 4 };
    log_probs = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
    const auto q3 = std::make_shared<const ov::genai::DraftProposal>(ov::genai::DraftProposal{{3, 4}, {0.75f, 0.25f}});
    const auto q4 = std::make_shared<const ov::genai::DraftProposal>(ov::genai::DraftProposal{{}, {0, 0, 0, 0, 1.f}});
    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs, 0, {}, nullptr, {nullptr, nullptr, nullptr, q3, q4}) }};
    ASSERT_EQ(m_pipeline.update_request(0, candidate_1, false).inserted_tokens_cnt, 2);

    auto proposals = m_pipeline.get_generated_requests().at(0).at(0).draft_proposals;
    ASSERT_EQ(proposals, (std::vector<ov::genai::DraftProposalPtr>{nullptr, nullptr, nullptr, q3, q4}));

    // A round without proposals leaves the pending ones in place.
    tokens.push_back(5);
    log_probs.push_back(0.6f);
    ov::genai::GeneratedSequences candidate_2{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    ASSERT_EQ(m_pipeline.update_request(0, candidate_2, false).inserted_tokens_cnt, 1);
    proposals = m_pipeline.get_generated_requests().at(0).at(0).draft_proposals;
    ASSERT_EQ(proposals, (std::vector<ov::genai::DraftProposalPtr>{nullptr, nullptr, nullptr, q3, q4, nullptr}));

    // Synchronizing a draft sequence with the main model's tokens commits them, so their proposals are dropped.
    ASSERT_EQ(m_pipeline.update_request(0, candidate_2, true).inserted_tokens_cnt, 0);
    EXPECT_TRUE(m_pipeline.get_generated_requests().at(0).at(0).draft_proposals.empty());
}

TEST_F(CBForSDTest, rejects_invalid_draft_proposal__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0 };
    std::vector<float> log_probs = { 0.1f };
    ov::genai::GeneratedSequences candidate{{ 0, ov::genai::GeneratedSequence(tokens, log_probs) }};
    ASSERT_EQ(m_pipeline.update_request(0, candidate, false).inserted_tokens_cnt, 1);

    tokens = { 0, 1 };
    log_probs = { 0.1f, 0.2f };
    const auto misaligned =
        std::make_shared<const ov::genai::DraftProposal>(ov::genai::DraftProposal{{1, 2}, {1.0f}});
    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs, 0, {}, nullptr, {nullptr, misaligned}) }};
    EXPECT_THROW(m_pipeline.update_request(0, candidate_1, false), ov::Exception);
}

TEST_F(CBForSDTest, dflash_candidate_update_without_logit_processor_update__one_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> target_seed_tokens = {10, 11};
    std::vector<float> target_seed_log_probs = {0.1f, 0.2f};
    ov::genai::GeneratedSequences target_seed{
        {0, ov::genai::GeneratedSequence(target_seed_tokens, target_seed_log_probs)}
    };
    auto update_result = m_pipeline.update_request(0, target_seed, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 2);

    std::vector<int64_t> draft_candidate_tokens = {10, 11, 20, 21, 22};
    std::vector<float> draft_candidate_log_probs = {0.0f, 0.0f, 0.3f, 0.4f, 0.5f};
    ov::genai::GeneratedSequences draft_candidate{
        {0, ov::genai::GeneratedSequence(draft_candidate_tokens, draft_candidate_log_probs)}
    };
    update_result = m_pipeline.update_request(0, draft_candidate, false);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    auto after_draft_update = m_pipeline.get_generated_requests();
    std::vector<float> expected_draft_log_probs = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    ASSERT_EQ(after_draft_update.at(0).at(0).token_ids, draft_candidate_tokens);
    ASSERT_EQ(after_draft_update.at(0).at(0).log_probs, expected_draft_log_probs);
    m_pipeline.register_generated_tokens(0, {20, 21, 22});

    std::vector<int64_t> validated_tokens = {10, 11, 20, 99};
    std::vector<float> validated_log_probs = {0.0f, 0.0f, 0.3f, 0.9f};
    ov::genai::GeneratedSequences validated_target{
        {0, ov::genai::GeneratedSequence(validated_tokens, validated_log_probs)}
    };
    update_result = m_pipeline.update_request(0, validated_target, false);
    ASSERT_EQ(update_result.removed_tokens_cnt, 2);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 1);

    auto after_validation_update = m_pipeline.get_generated_requests();
    std::vector<float> expected_validated_log_probs = {0.1f, 0.2f, 0.3f, 0.9f};
    ASSERT_EQ(after_validation_update.at(0).at(0).token_ids, validated_tokens);
    ASSERT_EQ(after_validation_update.at(0).at(0).log_probs, expected_validated_log_probs);
}

TEST_F(CBForSDTest, update_empty_sequence_by_not_empty__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens_0 = { 0, 1, 2 },
                         tokens_1 = { 0, 1 };
    std::vector<float> log_probs_0 = { 0.1f, 0.2f, 0.3f },
                       log_probs_1 = { 0.1f, 0.2f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens_0, log_probs_0) },
        { 1, ov::genai::GeneratedSequence(tokens_1, log_probs_1) }
    };
    
    auto before = m_pipeline.get_generated_requests();
    auto update_result = m_pipeline.update_request(0, candidate, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens_0);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs_0);

    ASSERT_EQ(after.at(0).size(), 1);
    ASSERT_EQ(after.at(0).size(), 1);
}

TEST_F(CBForSDTest, init_sequence_by_not_empty__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens_0 = { 0, 1, 2 },
                         tokens_1 = { 0, 1 };
    std::vector<float> log_probs_0 = { 0.1f, 0.2f, 0.3f },
                       log_probs_1 = { 0.1f, 0.2f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens_0, log_probs_0) },
        { 1, ov::genai::GeneratedSequence(tokens_1, log_probs_1) }
    };
    
    auto before = m_pipeline.get_generated_requests();
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 2);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens_1);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs_1);

    ASSERT_EQ(after.at(0).at(1).token_ids, tokens_1);
    ASSERT_EQ(after.at(0).at(1).log_probs, log_probs_1);
}

TEST_F(CBForSDTest, init_sequence_by_empty__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = {};
    std::vector<float> log_probs = {};
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(tokens, log_probs) },
    };
    
    auto before = m_pipeline.get_generated_requests();
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_EQ(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_EQ(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
    ASSERT_EQ(after.at(0).at(1).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(1).log_probs, log_probs);
}

TEST_F(CBForSDTest, no_updated_tokens__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens_0 = { 0, 1, 2 }, tokens_1 = { 0, 1 };
    std::vector<float> log_probs_0 = { 0.1f, 0.2f, 0.3f }, log_probs_1 = { 0.1f, 0.2f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens_0, log_probs_0) },
        { 1, ov::genai::GeneratedSequence(tokens_1, log_probs_1) },
    };
    
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 2);

    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens_1, log_probs_1) },
        { 1, ov::genai::GeneratedSequence(tokens_1, log_probs_1) },
    };

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens_1);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs_1);
    ASSERT_EQ(after.at(0).at(1).token_ids, tokens_1);
    ASSERT_EQ(after.at(0).at(1).log_probs, log_probs_1);
}

TEST_F(CBForSDTest, remove_tokens__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(tokens, log_probs) },
    };
    
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    std::vector<int64_t> tokens_new = { 0, 1 };
    std::vector<float> log_probs_new = { 0.1f, 0.2f };
    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(tokens_new, log_probs_new) },
    };

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 1);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 0);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_NE(after.at(0).at(1).token_ids, before.at(0).at(1).token_ids);
    ASSERT_NE(after.at(0).at(1).log_probs, before.at(0).at(1).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens_new);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs_new);
    ASSERT_EQ(after.at(0).at(1).token_ids, tokens_new);
    ASSERT_EQ(after.at(0).at(1).log_probs, log_probs_new);
}

TEST_F(CBForSDTest, remove_and_replace_tokens__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(tokens, log_probs) },
    };
    
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    std::vector<int64_t> new_tokens = { 0, 1, 4 };
    std::vector<float> new_log_probs = { 0.1f, 0.2f, 0.4f };
    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(new_tokens, new_log_probs) },
    };

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 1);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 1);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_EQ(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_EQ(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
    ASSERT_NE(after.at(0).at(1).token_ids, before.at(0).at(1).token_ids);
    ASSERT_NE(after.at(0).at(1).log_probs, before.at(0).at(1).log_probs);
    ASSERT_EQ(after.at(0).at(1).token_ids, new_tokens);
    ASSERT_EQ(after.at(0).at(1).log_probs, new_log_probs);
}

TEST_F(CBForSDTest, add_tokens__two_sequence) {
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    m_pipeline.add_request(0, input_tensor);

    std::vector<int64_t> tokens = { 0, 1, 2 };
    std::vector<float> log_probs = { 0.1f, 0.2f, 0.3f };
    ov::genai::GeneratedSequences candidate{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(tokens, log_probs) },
    };
    
    auto update_result = m_pipeline.init_request_by_candidate(0, candidate);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 3);

    tokens = { 0, 1, 2, 3, 4 };
    log_probs = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
    std::vector<int64_t> new_tokens = { 0, 1, 2, 3, 4, 5 };
    std::vector<float> new_log_probs = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f };
    ov::genai::GeneratedSequences candidate_1{
        { 0, ov::genai::GeneratedSequence(tokens, log_probs) },
        { 1, ov::genai::GeneratedSequence(new_tokens, new_log_probs) },
    };

    auto before = m_pipeline.get_generated_requests();
    update_result = m_pipeline.update_request(0, candidate_1, true);
    ASSERT_EQ(update_result.removed_tokens_cnt, 0);
    ASSERT_EQ(update_result.inserted_tokens_cnt, 2);

    auto after = m_pipeline.get_generated_requests();
    ASSERT_NE(after.at(0).at(0).token_ids, before.at(0).at(0).token_ids);
    ASSERT_NE(after.at(0).at(0).log_probs, before.at(0).at(0).log_probs);
    ASSERT_EQ(after.at(0).at(0).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(0).log_probs, log_probs);
    ASSERT_NE(after.at(0).at(1).token_ids, before.at(0).at(1).token_ids);
    ASSERT_NE(after.at(0).at(1).log_probs, before.at(0).at(1).log_probs);
    ASSERT_EQ(after.at(0).at(1).token_ids, tokens);
    ASSERT_EQ(after.at(0).at(1).log_probs, log_probs);
}
