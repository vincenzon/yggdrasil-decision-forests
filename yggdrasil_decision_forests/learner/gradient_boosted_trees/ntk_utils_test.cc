/*
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/ntk_utils.h"

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/learner/decision_tree/decision_tree.pb.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {
namespace {

using decision_tree::proto::DecisionTreeTrainingConfig;

TEST(NtkUtilsTest, ComputeCapacityBound) {
  // From Section 5.4 table:
  // Depth 1 (stumps): max_trees = n
  // Depth 2: max_trees = n/3
  // Depth 3: max_trees = n/7
  // Depth L: max_trees = n/(2^L - 1)

  // n = 1000, depth = 1: max_trees = 1000
  EXPECT_EQ(ComputeCapacityBound(1000, 1), 1000);

  // n = 1000, depth = 2: max_trees = 1000/3 = 333
  EXPECT_EQ(ComputeCapacityBound(1000, 2), 333);

  // n = 1000, depth = 3: max_trees = 1000/7 = 142
  EXPECT_EQ(ComputeCapacityBound(1000, 3), 142);

  // n = 1000, depth = 6: max_trees = 1000/63 = 15
  EXPECT_EQ(ComputeCapacityBound(1000, 6), 15);

  // n = 1000, depth = 10: max_trees = 1000/1023 = 0
  EXPECT_EQ(ComputeCapacityBound(1000, 10), 0);

  // Unlimited depth returns -1.
  EXPECT_EQ(ComputeCapacityBound(1000, -1), -1);
  EXPECT_EQ(ComputeCapacityBound(1000, 0), -1);
}

TEST(NtkUtilsTest, ApplyCapacityPolicyNone) {
  // CAPACITY_NONE should not modify num_trees.
  const int32_t num_trees =
      ApplyCapacityPolicy(1000, 6, 100, 1.0,
                          DecisionTreeTrainingConfig::CAPACITY_NONE);
  EXPECT_EQ(num_trees, 1000);
}

TEST(NtkUtilsTest, ApplyCapacityPolicyWarn) {
  // CAPACITY_WARN should not modify num_trees but should log a warning.
  // (We can't easily test logging, so just verify num_trees is unchanged.)
  const int32_t num_trees =
      ApplyCapacityPolicy(1000, 6, 100, 1.0,
                          DecisionTreeTrainingConfig::CAPACITY_WARN);
  EXPECT_EQ(num_trees, 1000);
}

TEST(NtkUtilsTest, ApplyCapacityPolicyClampTrees) {
  // n_effective = 100, depth = 6, leaves_per_tree = 63
  // max_trees = 100 / 63 = 1
  // Request 1000 trees, should clamp to 1.
  const int32_t num_trees =
      ApplyCapacityPolicy(1000, 6, 100, 1.0,
                          DecisionTreeTrainingConfig::CAPACITY_CLAMP_TREES);
  EXPECT_EQ(num_trees, 1);
}

TEST(NtkUtilsTest, ApplyCapacityPolicyClampTreesWithSubsample) {
  // n_training = 1000, subsample = 0.1, n_effective = 100
  // depth = 3, leaves_per_tree = 7
  // max_trees = 100 / 7 = 14
  // Request 1000 trees, should clamp to 14.
  const int32_t num_trees =
      ApplyCapacityPolicy(1000, 3, 1000, 0.1,
                          DecisionTreeTrainingConfig::CAPACITY_CLAMP_TREES);
  EXPECT_EQ(num_trees, 14);
}

TEST(NtkUtilsTest, ApplyCapacityPolicyWithinBound) {
  // n_effective = 1000, depth = 3, leaves_per_tree = 7
  // max_trees = 1000 / 7 = 142
  // Request 100 trees, should NOT clamp (100 <= 142).
  const int32_t num_trees =
      ApplyCapacityPolicy(100, 3, 1000, 1.0,
                          DecisionTreeTrainingConfig::CAPACITY_CLAMP_TREES);
  EXPECT_EQ(num_trees, 100);
}

TEST(NtkUtilsTest, ComputeClampedMaxDepth) {
  // n_effective = 100, num_trees = 10
  // max_leaves_per_tree = 100/10 + 1 = 11
  // max_depth = floor(log2(11)) = 3
  // If current max_depth = 6, should clamp to 3.
  const int32_t clamped_depth =
      ComputeClampedMaxDepth(10, 6, 100, 1.0);
  EXPECT_EQ(clamped_depth, 3);
}

TEST(NtkUtilsTest, ComputeClampedMaxDepthNoClampNeeded) {
  // n_effective = 10000, num_trees = 10
  // max_leaves_per_tree = 10000/10 + 1 = 1001
  // max_depth = floor(log2(1001)) = 9
  // If current max_depth = 6, should NOT clamp (6 < 9).
  const int32_t clamped_depth =
      ComputeClampedMaxDepth(10, 6, 10000, 1.0);
  EXPECT_EQ(clamped_depth, 6);
}

TEST(NtkUtilsTest, ComputeAdaptiveShrinkageOff) {
  // When adaptive shrinkage is OFF, should return the cap.
  decision_tree::DecisionTree tree;
  const double eta = ComputeAdaptiveShrinkage(
      tree, 1000, 10.0,
      proto::GradientBoostedTreesTrainingConfig::ADAPTIVE_SHRINKAGE_OFF, 0.1);
  EXPECT_DOUBLE_EQ(eta, 0.1);
}

TEST(NtkUtilsTest, ComputeAdaptiveShrinkageConservative) {
  // Conservative: eta = 1 / gain
  // gain = 10.0, eta = 1/10 = 0.1, cap = 0.5
  // Result should be min(0.1, 0.5) = 0.1
  decision_tree::DecisionTree tree;
  const double eta = ComputeAdaptiveShrinkage(
      tree, 1000, 10.0,
      proto::GradientBoostedTreesTrainingConfig::ADAPTIVE_SHRINKAGE_CONSERVATIVE,
      0.5);
  EXPECT_DOUBLE_EQ(eta, 0.1);
}

TEST(NtkUtilsTest, ComputeAdaptiveShrinkageConservativeClamped) {
  // Conservative: eta = 1 / gain
  // gain = 0.5, eta = 1/0.5 = 2.0, cap = 0.1
  // Result should be clamped to 0.1
  decision_tree::DecisionTree tree;
  const double eta = ComputeAdaptiveShrinkage(
      tree, 1000, 0.5,
      proto::GradientBoostedTreesTrainingConfig::ADAPTIVE_SHRINKAGE_CONSERVATIVE,
      0.1);
  EXPECT_DOUBLE_EQ(eta, 0.1);
}

}  // namespace
}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests
