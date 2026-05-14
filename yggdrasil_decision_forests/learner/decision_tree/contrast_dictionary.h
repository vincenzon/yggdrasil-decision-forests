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

// Contrast dictionary feature scoring utilities.
// Based on "The Neural Tangent Kernel in Random Forests and Gradient Boosted
// Models" (Litman & Guo, 2026), Section 5.5.
//
// The contrast dictionary for a feature f consists of vectors c_{f,k} that
// represent the difference between the mean label in the left and right
// children for each possible split threshold k:
//
//   c_{f,k} = mean(y|L_k) * 1_{L_k} - mean(y|R_k) * 1_{R_k}
//
// This module provides three feature scoring methods:
// 1. Individual signal (s_f): max squared norm across thresholds
// 2. Pairwise redundancy (G_fg^mean): average squared cosine between features
// 3. Subspace overlap: SVD-based measure of dictionary subspace alignment

#ifndef YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_CONTRAST_DICTIONARY_H_
#define YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_CONTRAST_DICTIONARY_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

// Represents a contrast vector for a single split threshold.
struct ContrastVector {
  // The feature index.
  int32_t feature_idx;
  // The threshold index (for numerical features) or category mask.
  int32_t threshold_idx;
  // Squared norm of the contrast vector: ||c||^2 = sum_i c_i^2
  double squared_norm;
  // Number of examples in the left child.
  int64_t left_count;
  // Number of examples in the right child.
  int64_t right_count;
  // Mean label in the left child.
  double left_mean;
  // Mean label in the right child.
  double right_mean;
};

// Holds the contrast dictionary for a single feature.
struct FeatureContrastDictionary {
  int32_t feature_idx;
  std::vector<ContrastVector> contrasts;

  // Individual signal score: max squared norm across all thresholds.
  double IndividualSignal() const {
    double max_sq_norm = 0.0;
    for (const auto& c : contrasts) {
      max_sq_norm = std::max(max_sq_norm, c.squared_norm);
    }
    return max_sq_norm;
  }
};

// Computes the contrast dictionary for a numerical feature.
// Returns a list of ContrastVector entries, one for each unique split threshold.
//
// Arguments:
//   dataset: The vertical dataset containing the feature and labels.
//   feature_idx: Index of the numerical feature.
//   label_idx: Index of the label column (must be numerical).
//   max_thresholds: Maximum number of thresholds to consider (for efficiency).
//                   If -1, considers all unique values.
//
// Returns:
//   A FeatureContrastDictionary containing contrast vectors for each threshold.
absl::StatusOr<FeatureContrastDictionary> ComputeNumericalContrastDictionary(
    const dataset::VerticalDataset& dataset, int32_t feature_idx,
    int32_t label_idx, int32_t max_thresholds = -1);

// Computes the contrast dictionary for a categorical feature.
// For categorical features, each contrast vector represents a binary split
// where one category goes left and the rest go right.
//
// Arguments:
//   dataset: The vertical dataset containing the feature and labels.
//   feature_idx: Index of the categorical feature.
//   label_idx: Index of the label column (must be numerical).
//
// Returns:
//   A FeatureContrastDictionary containing contrast vectors for each category.
absl::StatusOr<FeatureContrastDictionary> ComputeCategoricalContrastDictionary(
    const dataset::VerticalDataset& dataset, int32_t feature_idx,
    int32_t label_idx);

// Computes the pairwise redundancy between two features.
// G_fg^mean = (1 / |D_f| / |D_g|) * sum_{k,l} (c_f,k · c_g,l / ||c_f,k|| / ||c_g,l||)^2
//
// This measures the average squared cosine between all pairs of contrast
// vectors from the two features. High values indicate redundant features.
//
// Arguments:
//   dict_f: Contrast dictionary for feature f.
//   dict_g: Contrast dictionary for feature g.
//   dataset: The dataset (needed to compute dot products).
//   label_idx: Index of the label column.
//
// Returns:
//   The pairwise redundancy score in [0, 1].
double PairwiseRedundancyMean(const FeatureContrastDictionary& dict_f,
                              const FeatureContrastDictionary& dict_g,
                              const dataset::VerticalDataset& dataset,
                              int32_t label_idx);

// Computes all contrast dictionaries for a dataset.
// Only considers numerical and categorical features that are valid for
// prediction (i.e., not the label or other special columns).
//
// Arguments:
//   dataset: The vertical dataset.
//   label_idx: Index of the label column.
//   input_features: List of feature indices to consider.
//   max_thresholds: Maximum thresholds per numerical feature.
//
// Returns:
//   A vector of FeatureContrastDictionary, one per input feature.
absl::StatusOr<std::vector<FeatureContrastDictionary>>
ComputeAllContrastDictionaries(const dataset::VerticalDataset& dataset,
                               int32_t label_idx,
                               const std::vector<int32_t>& input_features,
                               int32_t max_thresholds = 255);

// Computes the individual signal score for all features.
// This is s_f = max_k ||c_{f,k}||^2 for each feature f.
//
// Arguments:
//   dictionaries: The contrast dictionaries for all features.
//
// Returns:
//   A vector of (feature_idx, individual_signal) pairs, sorted by descending
//   signal strength.
std::vector<std::pair<int32_t, double>> ComputeAllIndividualSignals(
    const std::vector<FeatureContrastDictionary>& dictionaries);

// Computes the pairwise redundancy matrix for all features.
//
// Arguments:
//   dictionaries: The contrast dictionaries for all features.
//   dataset: The dataset.
//   label_idx: Index of the label column.
//
// Returns:
//   A symmetric matrix where entry (i,j) is the pairwise redundancy between
//   features dictionaries[i].feature_idx and dictionaries[j].feature_idx.
std::vector<std::vector<double>> ComputePairwiseRedundancyMatrix(
    const std::vector<FeatureContrastDictionary>& dictionaries,
    const dataset::VerticalDataset& dataset, int32_t label_idx);

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests

#endif  // YGGDRASIL_DECISION_FORESTS_LEARNER_DECISION_TREE_CONTRAST_DICTIONARY_H_
