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

#include "yggdrasil_decision_forests/learner/decision_tree/contrast_dictionary.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {

namespace {

// Helper to compute sum and count for labels in a subset of examples.
struct LabelStats {
  double sum = 0.0;
  int64_t count = 0;

  void Add(double value) {
    sum += value;
    count++;
  }

  double Mean() const { return count > 0 ? sum / count : 0.0; }
};

}  // namespace

absl::StatusOr<FeatureContrastDictionary> ComputeNumericalContrastDictionary(
    const dataset::VerticalDataset& dataset, int32_t feature_idx,
    int32_t label_idx, int32_t max_thresholds) {
  FeatureContrastDictionary result;
  result.feature_idx = feature_idx;

  // Get the feature and label columns.
  const auto* feature_col =
      dataset.ColumnWithCastOrNull<dataset::VerticalDataset::NumericalColumn>(
          feature_idx);
  const auto* label_col =
      dataset.ColumnWithCastOrNull<dataset::VerticalDataset::NumericalColumn>(
          label_idx);

  if (feature_col == nullptr) {
    return absl::InvalidArgumentError(
        "Feature column is not numerical or does not exist.");
  }
  if (label_col == nullptr) {
    return absl::InvalidArgumentError(
        "Label column is not numerical or does not exist.");
  }

  const auto& features = feature_col->values();
  const auto& labels = label_col->values();
  const int64_t n = dataset.nrow();

  if (n == 0) {
    return result;
  }

  // Create sorted indices by feature value.
  std::vector<int64_t> sorted_indices(n);
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&features](int64_t a, int64_t b) {
              return features[a] < features[b];
            });

  // Compute total label statistics.
  double total_sum = 0.0;
  for (int64_t i = 0; i < n; i++) {
    total_sum += labels[i];
  }
  const double total_mean = total_sum / n;

  // Find unique threshold values.
  std::vector<std::pair<float, int64_t>> thresholds;  // (value, split_position)
  for (int64_t i = 0; i < n - 1; i++) {
    const int64_t idx = sorted_indices[i];
    const int64_t next_idx = sorted_indices[i + 1];
    if (features[idx] < features[next_idx]) {
      thresholds.push_back({features[idx], i + 1});
    }
  }

  // Subsample thresholds if too many.
  if (max_thresholds > 0 &&
      static_cast<int32_t>(thresholds.size()) > max_thresholds) {
    std::vector<std::pair<float, int64_t>> sampled;
    const double step =
        static_cast<double>(thresholds.size()) / max_thresholds;
    for (int i = 0; i < max_thresholds; i++) {
      sampled.push_back(thresholds[static_cast<int>(i * step)]);
    }
    thresholds = std::move(sampled);
  }

  // Compute contrast vectors for each threshold.
  result.contrasts.reserve(thresholds.size());

  // Running statistics for left side.
  double left_sum = 0.0;
  int64_t left_idx = 0;

  for (int32_t t = 0; t < static_cast<int32_t>(thresholds.size()); t++) {
    const auto& [threshold_value, split_pos] = thresholds[t];

    // Update left statistics up to split position.
    while (left_idx < split_pos) {
      const int64_t example_idx = sorted_indices[left_idx];
      left_sum += labels[example_idx];
      left_idx++;
    }

    const int64_t left_count = split_pos;
    const int64_t right_count = n - split_pos;
    const double left_mean = left_count > 0 ? left_sum / left_count : 0.0;
    const double right_mean =
        right_count > 0 ? (total_sum - left_sum) / right_count : 0.0;

    // Squared norm of contrast vector:
    // ||c||^2 = sum_i c_i^2
    // where c_i = left_mean if i in left, -right_mean if i in right
    // = left_count * left_mean^2 + right_count * right_mean^2
    const double squared_norm =
        left_count * left_mean * left_mean +
        right_count * right_mean * right_mean;

    ContrastVector cv;
    cv.feature_idx = feature_idx;
    cv.threshold_idx = t;
    cv.squared_norm = squared_norm;
    cv.left_count = left_count;
    cv.right_count = right_count;
    cv.left_mean = left_mean;
    cv.right_mean = right_mean;

    result.contrasts.push_back(cv);
  }

  return result;
}

absl::StatusOr<FeatureContrastDictionary> ComputeCategoricalContrastDictionary(
    const dataset::VerticalDataset& dataset, int32_t feature_idx,
    int32_t label_idx) {
  FeatureContrastDictionary result;
  result.feature_idx = feature_idx;

  const auto* feature_col =
      dataset.ColumnWithCastOrNull<dataset::VerticalDataset::CategoricalColumn>(
          feature_idx);
  const auto* label_col =
      dataset.ColumnWithCastOrNull<dataset::VerticalDataset::NumericalColumn>(
          label_idx);

  if (feature_col == nullptr) {
    return absl::InvalidArgumentError(
        "Feature column is not categorical or does not exist.");
  }
  if (label_col == nullptr) {
    return absl::InvalidArgumentError(
        "Label column is not numerical or does not exist.");
  }

  const auto& features = feature_col->values();
  const auto& labels = label_col->values();
  const int64_t n = dataset.nrow();

  if (n == 0) {
    return result;
  }

  // Find max category value.
  int32_t max_category = 0;
  for (int64_t i = 0; i < n; i++) {
    max_category = std::max(max_category, features[i]);
  }

  // Compute statistics per category.
  std::vector<LabelStats> category_stats(max_category + 1);
  double total_sum = 0.0;
  for (int64_t i = 0; i < n; i++) {
    const int32_t cat = features[i];
    if (cat >= 0 && cat <= max_category) {
      category_stats[cat].Add(labels[i]);
      total_sum += labels[i];
    }
  }

  // Create contrast vector for each category (one-vs-rest split).
  for (int32_t cat = 0; cat <= max_category; cat++) {
    if (category_stats[cat].count == 0 ||
        category_stats[cat].count == n) {
      continue;  // Skip empty or full categories.
    }

    const int64_t left_count = category_stats[cat].count;
    const int64_t right_count = n - left_count;
    const double left_mean = category_stats[cat].Mean();
    const double right_mean =
        (total_sum - category_stats[cat].sum) / right_count;

    const double squared_norm =
        left_count * left_mean * left_mean +
        right_count * right_mean * right_mean;

    ContrastVector cv;
    cv.feature_idx = feature_idx;
    cv.threshold_idx = cat;
    cv.squared_norm = squared_norm;
    cv.left_count = left_count;
    cv.right_count = right_count;
    cv.left_mean = left_mean;
    cv.right_mean = right_mean;

    result.contrasts.push_back(cv);
  }

  return result;
}

double PairwiseRedundancyMean(const FeatureContrastDictionary& dict_f,
                              const FeatureContrastDictionary& dict_g,
                              const dataset::VerticalDataset& dataset,
                              int32_t label_idx) {
  // This is a simplified implementation that uses the contrast vector
  // statistics rather than computing full dot products.
  // For exact computation, we would need to reconstruct the full vectors.

  if (dict_f.contrasts.empty() || dict_g.contrasts.empty()) {
    return 0.0;
  }

  // Approximate: use the correlation between contrast vectors based on
  // their squared norms and overlap in label statistics.
  // This is a heuristic approximation.

  double sum_squared_cosine = 0.0;
  int count = 0;

  for (const auto& cf : dict_f.contrasts) {
    for (const auto& cg : dict_g.contrasts) {
      if (cf.squared_norm <= 0 || cg.squared_norm <= 0) {
        continue;
      }

      // Approximation: assume the dot product is proportional to
      // the similarity of the split patterns.
      // A proper implementation would require full vector reconstruction.
      const double norm_f = std::sqrt(cf.squared_norm);
      const double norm_g = std::sqrt(cg.squared_norm);

      // Estimate dot product using the means and counts.
      // This is an approximation that works when features are correlated.
      const double approx_dot =
          cf.left_count * cf.left_mean * cg.left_mean *
              std::min(cf.left_count, cg.left_count) /
              std::max(cf.left_count, cg.left_count) +
          cf.right_count * (-cf.right_mean) * (-cg.right_mean) *
              std::min(cf.right_count, cg.right_count) /
              std::max(cf.right_count, cg.right_count);

      const double cosine = approx_dot / (norm_f * norm_g);
      const double clamped_cosine = std::max(-1.0, std::min(1.0, cosine));
      sum_squared_cosine += clamped_cosine * clamped_cosine;
      count++;
    }
  }

  return count > 0 ? sum_squared_cosine / count : 0.0;
}

absl::StatusOr<std::vector<FeatureContrastDictionary>>
ComputeAllContrastDictionaries(const dataset::VerticalDataset& dataset,
                               int32_t label_idx,
                               const std::vector<int32_t>& input_features,
                               int32_t max_thresholds) {
  std::vector<FeatureContrastDictionary> result;
  result.reserve(input_features.size());

  for (int32_t feature_idx : input_features) {
    if (feature_idx == label_idx) {
      continue;
    }

    const auto col_type = dataset.column(feature_idx)->type();

    absl::StatusOr<FeatureContrastDictionary> dict;
    if (col_type == dataset::proto::ColumnType::NUMERICAL) {
      dict = ComputeNumericalContrastDictionary(dataset, feature_idx, label_idx,
                                                 max_thresholds);
    } else if (col_type == dataset::proto::ColumnType::CATEGORICAL) {
      dict = ComputeCategoricalContrastDictionary(dataset, feature_idx,
                                                   label_idx);
    } else {
      // Skip unsupported column types.
      continue;
    }

    if (dict.ok()) {
      result.push_back(std::move(*dict));
    }
  }

  return result;
}

std::vector<std::pair<int32_t, double>> ComputeAllIndividualSignals(
    const std::vector<FeatureContrastDictionary>& dictionaries) {
  std::vector<std::pair<int32_t, double>> signals;
  signals.reserve(dictionaries.size());

  for (const auto& dict : dictionaries) {
    signals.push_back({dict.feature_idx, dict.IndividualSignal()});
  }

  // Sort by descending signal strength.
  std::sort(signals.begin(), signals.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  return signals;
}

std::vector<std::vector<double>> ComputePairwiseRedundancyMatrix(
    const std::vector<FeatureContrastDictionary>& dictionaries,
    const dataset::VerticalDataset& dataset, int32_t label_idx) {
  const size_t n = dictionaries.size();
  std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));

  for (size_t i = 0; i < n; i++) {
    matrix[i][i] = 1.0;  // Self-redundancy is 1.
    for (size_t j = i + 1; j < n; j++) {
      const double redundancy =
          PairwiseRedundancyMean(dictionaries[i], dictionaries[j], dataset,
                                 label_idx);
      matrix[i][j] = redundancy;
      matrix[j][i] = redundancy;
    }
  }

  return matrix;
}

}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
