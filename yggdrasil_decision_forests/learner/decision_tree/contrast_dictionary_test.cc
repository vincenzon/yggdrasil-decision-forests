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

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "yggdrasil_decision_forests/dataset/data_spec.pb.h"
#include "yggdrasil_decision_forests/dataset/vertical_dataset.h"
#include "yggdrasil_decision_forests/utils/test.h"

namespace yggdrasil_decision_forests {
namespace model {
namespace decision_tree {
namespace {

// Helper to create a simple dataset for testing.
dataset::VerticalDataset CreateSimpleDataset() {
  dataset::proto::DataSpecification data_spec;

  // Feature column.
  auto* col_feature = data_spec.add_columns();
  col_feature->set_name("feature");
  col_feature->set_type(dataset::proto::ColumnType::NUMERICAL);

  // Label column.
  auto* col_label = data_spec.add_columns();
  col_label->set_name("label");
  col_label->set_type(dataset::proto::ColumnType::NUMERICAL);

  dataset::VerticalDataset dataset;
  dataset.set_data_spec(data_spec);
  EXPECT_OK(dataset.CreateColumnsFromDataspec());

  return dataset;
}

TEST(ContrastDictionaryTest, NumericalContrastDictionaryBasic) {
  auto dataset = CreateSimpleDataset();

  // Add some examples: feature = [1, 2, 3, 4], label = [0, 0, 1, 1]
  // This should create a clear split at feature = 2.5.
  auto* feature_col =
      dataset.MutableColumnWithCastOrNull<dataset::VerticalDataset::NumericalColumn>(0);
  auto* label_col =
      dataset.MutableColumnWithCastOrNull<dataset::VerticalDataset::NumericalColumn>(1);

  feature_col->Add(1.0f);
  label_col->Add(0.0f);
  feature_col->Add(2.0f);
  label_col->Add(0.0f);
  feature_col->Add(3.0f);
  label_col->Add(1.0f);
  feature_col->Add(4.0f);
  label_col->Add(1.0f);

  dataset.set_nrow(4);

  auto dict_result = ComputeNumericalContrastDictionary(dataset, 0, 1);
  ASSERT_TRUE(dict_result.ok());

  const auto& dict = *dict_result;
  EXPECT_EQ(dict.feature_idx, 0);
  EXPECT_GT(dict.contrasts.size(), 0);

  // The best split should be at threshold 2 (between 2 and 3).
  // Left: [0, 0], mean = 0
  // Right: [1, 1], mean = 1
  // Squared norm = 2*0^2 + 2*1^2 = 2
  bool found_optimal_split = false;
  for (const auto& cv : dict.contrasts) {
    if (cv.left_count == 2 && cv.right_count == 2) {
      EXPECT_DOUBLE_EQ(cv.left_mean, 0.0);
      EXPECT_DOUBLE_EQ(cv.right_mean, 1.0);
      EXPECT_DOUBLE_EQ(cv.squared_norm, 2.0);
      found_optimal_split = true;
    }
  }
  EXPECT_TRUE(found_optimal_split);
}

TEST(ContrastDictionaryTest, IndividualSignal) {
  auto dataset = CreateSimpleDataset();

  // Feature with strong signal.
  auto* feature_col =
      dataset.MutableColumnWithCastOrNull<dataset::VerticalDataset::NumericalColumn>(0);
  auto* label_col =
      dataset.MutableColumnWithCastOrNull<dataset::VerticalDataset::NumericalColumn>(1);

  for (int i = 0; i < 10; i++) {
    feature_col->Add(static_cast<float>(i));
    label_col->Add(i < 5 ? 0.0f : 10.0f);
  }
  dataset.set_nrow(10);

  auto dict_result = ComputeNumericalContrastDictionary(dataset, 0, 1);
  ASSERT_TRUE(dict_result.ok());

  const double signal = dict_result->IndividualSignal();
  EXPECT_GT(signal, 0.0);
}

TEST(ContrastDictionaryTest, EmptyDataset) {
  auto dataset = CreateSimpleDataset();
  dataset.set_nrow(0);

  auto dict_result = ComputeNumericalContrastDictionary(dataset, 0, 1);
  ASSERT_TRUE(dict_result.ok());
  EXPECT_TRUE(dict_result->contrasts.empty());
}

TEST(ContrastDictionaryTest, SingleExample) {
  auto dataset = CreateSimpleDataset();

  auto* feature_col =
      dataset.MutableColumnWithCastOrNull<dataset::VerticalDataset::NumericalColumn>(0);
  auto* label_col =
      dataset.MutableColumnWithCastOrNull<dataset::VerticalDataset::NumericalColumn>(1);

  feature_col->Add(1.0f);
  label_col->Add(0.0f);
  dataset.set_nrow(1);

  auto dict_result = ComputeNumericalContrastDictionary(dataset, 0, 1);
  ASSERT_TRUE(dict_result.ok());
  // Cannot split with single example.
  EXPECT_TRUE(dict_result->contrasts.empty());
}

TEST(ContrastDictionaryTest, ComputeAllIndividualSignals) {
  FeatureContrastDictionary dict1;
  dict1.feature_idx = 0;
  dict1.contrasts.push_back({0, 0, 10.0, 5, 5, 0.0, 1.0});
  dict1.contrasts.push_back({0, 1, 5.0, 3, 7, 0.0, 0.5});

  FeatureContrastDictionary dict2;
  dict2.feature_idx = 1;
  dict2.contrasts.push_back({1, 0, 20.0, 5, 5, 0.0, 2.0});

  std::vector<FeatureContrastDictionary> dicts = {dict1, dict2};

  auto signals = ComputeAllIndividualSignals(dicts);

  ASSERT_EQ(signals.size(), 2);
  // dict2 has higher signal (20.0 vs 10.0), so it should come first.
  EXPECT_EQ(signals[0].first, 1);
  EXPECT_DOUBLE_EQ(signals[0].second, 20.0);
  EXPECT_EQ(signals[1].first, 0);
  EXPECT_DOUBLE_EQ(signals[1].second, 10.0);
}

}  // namespace
}  // namespace decision_tree
}  // namespace model
}  // namespace yggdrasil_decision_forests
