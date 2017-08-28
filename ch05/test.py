# File: test.py
#    from chapter 5 of _Tree-based Machine Learning Algorithms_
#
# Author: Clinton Sheppard <fluentcoder@gmail.com>
# Copyright (c) 2017 Clinton Sheppard
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.

from collections import Counter
from forest import Forest
import dtree


data = dtree.read_csv('..\ch04\census.csv')
continuousColumns = ['Age']
data = dtree.prepare_data(data, continuousColumns)
outcomeLabel = 'Born'

forest = Forest(data, outcomeLabel, continuousColumns)
testData = ['Elizabeth', 'female', 'Married', 16, 'Daughter']
predicted = forest.get_prediction(testData)
print("predicted: {}".format(predicted))

forest = Forest(data, outcomeLabel, continuousColumns)
predictions = []
for _ in range(0, 100):
    predictions.append(forest.get_prediction(testData))
    forest.populate()
counts = Counter(predictions)
print("predictions: {}".format(counts.most_common()))