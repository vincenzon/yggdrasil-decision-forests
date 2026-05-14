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

// Utilities for NTK-theory based improvements to gradient boosted trees.
// Based on "The Neural Tangent Kernel in Random Forests and Gradient Boosted
// Models" (Litman & Guo, 2026).

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_NTK_UTILS_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_NTK_UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "absl/log/log.h"
#include "yggdrasil_decision_forests/learner/gradient_boosted_trees/gradient_boosted_trees.pb.h"
#include "yggdrasil_decision_forests/model/decision_tree/decision_tree.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace gradient_boosted_trees {

// Computes the capacity bound: max_trees = n_effective / (2^max_depth - 1)
// Returns -1 if max_depth is unlimited (-1) or if the bound cannot be computed.
inline int64_t ComputeCapacityBound(int64_t n_effective, int32_t max_depth) {
  if (max_depth < 1) {
    return -1;  // Unlimited depth, cannot compute bound
  }
  const int64_t leaves_per_tree = (1LL << max_depth) - 1;
  if (leaves_per_tree <= 0) {
    return -1;
  }
  return n_effective / leaves_per_tree;
}

// Checks the capacity constraint and applies the configured policy.
// Returns the (potentially modified) num_trees value.
// Logs warnings or errors as appropriate based on the policy.
inline int32_t ApplyCapacityPolicy(
    int32_t num_trees, int32_t max_depth, int64_t n_training,
    double subsample_ratio,
    decision_tree::proto::DecisionTreeTrainingConfig::CapacityPolicy policy) {
  using DTConfig = decision_tree::proto::DecisionTreeTrainingConfig;

  if (policy == DTConfig::CAPACITY_NONE) {
    return num_trees;
  }

  // Cannot compute bound with unlimited depth
  if (max_depth < 1) {
    if (policy == DTConfig::CAPACITY_CLAMP_TREES ||
        policy == DTConfig::CAPACITY_CLAMP_DEPTH) {
      LOG(WARNING) << "Capacity policy CLAMP_TREES/CLAMP_DEPTH requires "
                      "finite max_depth. Ignoring policy.";
    }
    return num_trees;
  }

  const int64_t n_effective =
      static_cast<int64_t>(n_training * subsample_ratio);
  const int64_t leaves_per_tree = (1LL << max_depth) - 1;
  const int64_t total_capacity = static_cast<int64_t>(num_trees) * leaves_per_tree;

  if (total_capacity <= n_effective) {
    // Within capacity bound, no action needed
    return num_trees;
  }

  // Capacity bound exceeded
  const int64_t max_trees = ComputeCapacityBound(n_effective, max_depth);

  switch (policy) {
    case DTConfig::CAPACITY_WARN:
      LOG(WARNING) << "Capacity bound exceeded: T * (2^L - 1) = "
                   << total_capacity << " > n_effective = " << n_effective
                   << " (num_trees=" << num_trees << ", max_depth=" << max_depth
                   << ", n_training=" << n_training
                   << ", subsample=" << subsample_ratio << "). "
                   << "Consider reducing num_trees to " << max_trees
                   << " or max_depth to avoid overfitting. "
                   << "See NTK theory (Litman & Guo, 2026) Section 5.4.";
      return num_trees;

    case DTConfig::CAPACITY_CLAMP_TREES:
      if (max_trees < num_trees) {
        LOG(INFO) << "Clamping num_trees from " << num_trees << " to "
                  << max_trees << " to satisfy capacity bound T * (2^L - 1) <= "
                  << n_effective << ".";
        return static_cast<int32_t>(std::max(max_trees, static_cast<int64_t>(1)));
      }
      return num_trees;

    case DTConfig::CAPACITY_CLAMP_DEPTH:
      // This case should modify max_depth, not num_trees.
      // Caller should handle this separately.
      LOG(WARNING) << "CAPACITY_CLAMP_DEPTH should be handled by caller.";
      return num_trees;

    default:
      return num_trees;
  }
}

// Computes the max_depth needed to satisfy capacity constraint.
// Returns the (potentially reduced) max_depth.
inline int32_t ComputeClampedMaxDepth(int32_t num_trees, int32_t max_depth,
                                       int64_t n_training,
                                       double subsample_ratio) {
  if (max_depth < 1 || num_trees <= 0) {
    return max_depth;
  }

  const int64_t n_effective =
      static_cast<int64_t>(n_training * subsample_ratio);

  // Find max_depth such that num_trees * (2^max_depth - 1) <= n_effective
  // => 2^max_depth <= n_effective / num_trees + 1
  // => max_depth <= log2(n_effective / num_trees + 1)
  const double max_leaves_per_tree =
      static_cast<double>(n_effective) / num_trees + 1.0;
  const int32_t clamped_depth =
      static_cast<int32_t>(std::floor(std::log2(max_leaves_per_tree)));

  if (clamped_depth < max_depth && clamped_depth >= 1) {
    LOG(INFO) << "Clamping max_depth from " << max_depth << " to "
              << clamped_depth
              << " to satisfy capacity bound T * (2^L - 1) <= " << n_effective
              << ".";
    return clamped_depth;
  }
  return max_depth;
}

// Computes the adaptive shrinkage (learning rate) for a tree based on its
// leaf statistics. From NTK theory (Litman & Guo, 2026) Section 5.3.
//
// Arguments:
//   tree: The decision tree to compute shrinkage for.
//   n_training: Total number of training examples.
//   total_gain: Sum of gains across all splits in the tree.
//   policy: Which adaptive shrinkage formula to use.
//   cap: Upper bound on the computed shrinkage.
//
// Returns the computed shrinkage value, clamped to [0, cap].
inline double ComputeAdaptiveShrinkage(
    const decision_tree::DecisionTree& tree, int64_t n_training,
    double total_gain,
    proto::GradientBoostedTreesTrainingConfig::AdaptiveShrinkage policy,
    double cap) {
  using GBTConfig = proto::GradientBoostedTreesTrainingConfig;

  if (policy == GBTConfig::ADAPTIVE_SHRINKAGE_OFF) {
    return cap;  // Use constant shrinkage (the cap acts as the constant)
  }

  double eta = cap;

  switch (policy) {
    case GBTConfig::ADAPTIVE_SHRINKAGE_EXACT: {
      // eta_t = n / max_l(v_l^2 * n_l)
      // Where v_l is the leaf value and n_l is the number of examples in leaf l.
      double max_v2_n = 0.0;
      tree.IterateOnNodes(
          [&max_v2_n](const decision_tree::NodeWithChildren& node,
                      const int depth) {
            if (node.IsLeaf()) {
              const double v = node.node().regressor().top_value();
              const double n_l = static_cast<double>(
                  node.node().num_pos_training_examples_without_weight());
              const double v2_n = v * v * n_l;
              max_v2_n = std::max(max_v2_n, v2_n);
            }
          });
      if (max_v2_n > 0.0) {
        eta = static_cast<double>(n_training) / max_v2_n;
      }
    } break;

    case GBTConfig::ADAPTIVE_SHRINKAGE_CONSERVATIVE: {
      // eta_t = 1 / gain_t
      if (total_gain > 0.0) {
        eta = 1.0 / total_gain;
      }
    } break;

    case GBTConfig::ADAPTIVE_SHRINKAGE_OPTIMISTIC: {
      // eta_t = L_t / gain_t
      // Where L_t is the number of leaves in the tree.
      if (total_gain > 0.0) {
        int num_leaves = 0;
        tree.IterateOnNodes(
            [&num_leaves](const decision_tree::NodeWithChildren& node,
                          const int depth) {
              if (node.IsLeaf()) {
                num_leaves++;
              }
            });
        eta = static_cast<double>(num_leaves) / total_gain;
      }
    } break;

    default:
      break;
  }

  // Clamp to [0, cap]
  return std::min(std::max(eta, 0.0), cap);
}

// Checks if a tree has any meaningful splits (non-trivial).
// A trivial tree is one where the root is a leaf (no splits made).
// Used for signal channel saturation stopping criterion.
inline bool TreeHasMeaningfulSplits(const decision_tree::DecisionTree& tree) {
  const auto& root = tree.root();
  return !root.IsLeaf();
}

}  // namespace gradient_boosted_trees
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_GRADIENT_BOOSTED_TREES_NTK_UTILS_H_
