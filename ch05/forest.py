# File: forest.py
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
import dtree
import math
import random
import statistics


class Forest:
    def __init__(self, data, outcomeLabel, continuousAttributes=None,
                 dataRowIndexes=None, columnsNamesToIgnore=None):
        self.data = data
        self.outcomeLabel = outcomeLabel
        self.continuousAttributes = continuousAttributes \
            if columnsNamesToIgnore is None \
            else [i for i in continuousAttributes if
                  i not in columnsNamesToIgnore]
        self.numRows = math.ceil(math.sqrt(
            len(data) if dataRowIndexes is None else len(dataRowIndexes)))
        self.outcomeIndex = data[0].index(outcomeLabel)
        columnIdsToIgnore = set() if columnsNamesToIgnore is None else set(
            data[0].index(s) for s in columnsNamesToIgnore)
        columnIdsToIgnore.add(self.outcomeIndex)
        self.attrIndexesExceptOutcomeIndex = [i for i in range(0, len(data[0]))
                                              if i not in columnIdsToIgnore]
        self.numAttributes = math.ceil(
            math.sqrt(len(self.attrIndexesExceptOutcomeIndex)))
        self.dataRowIndexes = range(1, len(
            data)) if dataRowIndexes is None else dataRowIndexes
        self.numTrees = 200
        self.populate()

    def _build_tree(self):
        return dtree.build(self.data, self.outcomeLabel,
                           continuousAttributes=self.continuousAttributes,
                           dataIndexes={i for i in random.sample(
                               self.dataRowIndexes, self.numRows)},
                           attrIndexes=[
                               i for i in random.sample(
                                   self.attrIndexesExceptOutcomeIndex,
                                   self.numAttributes)])

    def populate(self):
        self._trees = [self._build_tree() for _ in range(0, self.numTrees)]

    def get_prediction(self, dataItem):
        sorted_predictions = self._get_predictions(dataItem)
        return sorted_predictions[0][0]

    def _get_predictions(self, dataItem):
        predictions = [t.get_prediction(dataItem) for t in self._trees]
        return Counter(p for p in predictions).most_common()


class Benchmark:
    @staticmethod
    def run(f):
        results = []
        for i in range(100):
            result = f()
            results.append(result)
            if i < 10 or i % 10 == 9:
                mean = statistics.mean(results)
                print("{} {:3.2f} {:3.2f}".format(
                    1 + i, mean,
                    statistics.stdev(results, mean) if i > 1 else 0))