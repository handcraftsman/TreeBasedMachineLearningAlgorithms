# File: titanic.py
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

from forest import Forest
from forest import Benchmark
import dtree
import random


continuousColumns = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
data = dtree.read_csv('train.csv')
data = dtree.prepare_data(data, continuousColumns)
outcomeLabel = 'Survived'
columnsToIgnore = ['PassengerId', 'Name', 'Ticket', 'Cabin']


def predict():
    trainingRowIds = random.sample(range(1, len(data)), int(.8 * len(data)))
    forest = Forest(data, outcomeLabel, continuousColumns, trainingRowIds, columnsToIgnore)
    correct = sum(1 for rowId, row in enumerate(data) if
                  rowId > 0 and
                  rowId not in trainingRowIds and
                  forest.get_prediction(row) == row[1])
    return 100 * correct / (len(data) - 1 - len(trainingRowIds))

Benchmark.run(predict)