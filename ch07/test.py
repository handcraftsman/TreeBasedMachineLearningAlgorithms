# File: test.py
#    from chapter 7 of _Tree-based Machine Learning Algorithms_
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

import dtree
import random
import forest

data = dtree.read_csv('mushrooms.csv')
outcomeLabel = 'class'
outcomeLabelIndex = data[0].index(outcomeLabel)
continuousAttributes = []


print("-- decision tree")
def predict():
    trainingRowIds = random.sample(range(1, len(data)),
                                   int(.01 * len(data)))
    tree = dtree.build(data, outcomeLabel, continuousAttributes,
                       dataIndexes=trainingRowIds)
    correct = sum(1 for rowId, row in enumerate(data) if
                  rowId > 0 and
                  rowId not in trainingRowIds and
                  tree.get_prediction(row) == row[outcomeLabelIndex])
    return 100 * correct / (len(data) - 1 - len(trainingRowIds))

forest.Benchmark.run(predict)


print("-- random forest")
def predict2():
    trainingRowIds = random.sample(range(1, len(data)),
                                   int(.01 * len(data)))
    f = forest.Forest(data, outcomeLabel, continuousAttributes,
                      trainingRowIds)
    correct = sum(1 for rowId, row in enumerate(data) if
                  rowId > 0 and
                  rowId not in trainingRowIds and
                  f.get_prediction(row) == row[outcomeLabelIndex])
    return 100 * correct / (len(data) - 1 - len(trainingRowIds))

forest.Benchmark.run(predict2)


print("-- boosted random forest")
def predict3():
    trainingRowIds = random.sample(range(1, len(data)),
                                   int(.01 * len(data)))
    f = forest.Forest(data, outcomeLabel, continuousAttributes,
                      trainingRowIds, boost=True)

    correct = sum(1 for rowId, row in enumerate(data) if
                  rowId > 0 and
                  rowId not in trainingRowIds and
                  f.get_prediction(row) == row[outcomeLabelIndex])
    return 100 * correct / (len(data) - 1 - len(trainingRowIds))

forest.Benchmark.run(predict3)
