# File: forest.py
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
import math
import random
import statistics
import operator

class Forest:
    def __init__(self, data, outcomeLabel, continuousAttributes=None,
                 dataRowIndexes=None, columnsNamesToIgnore=None,
                 boost=False):
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
        self.boost = boost
        self.weights = [.5 for _ in range(0, self.numTrees)]
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

        if not self.boost:
            return

        outcomeLabelIndex = self.data[0].index(self.outcomeLabel)
        anyChanged = True
        roundsRemaining = 10
        while anyChanged and roundsRemaining > 0:
            anyChanged = False
            roundsRemaining -= 1
            for dataRowIndex in self.dataRowIndexes:
                dataRow = self.data[dataRowIndex]
                sorted_predictions, predictions = self._get_predictions(
                    dataRow)
                expectedPrediction = dataRow[outcomeLabelIndex]
                if expectedPrediction == sorted_predictions[0][0]:
                    continue
                anyChanged = True
                actualPrediction = sorted_predictions[0][0]
                lookup = dict(sorted_predictions)
                expectedPredictionSum = lookup.get(expectedPrediction)
                difference = sorted_predictions[0][1] if \
                    expectedPredictionSum is None else \
                    sorted_predictions[0][1] - expectedPredictionSum
                maxDifference = difference / len(self.dataRowIndexes)
                if maxDifference == 0:
                    maxDifference = .5 / len(self.dataRowIndexes)
                for index, p in enumerate(predictions):
                    if p == expectedPrediction:
                        self.weights[index] = min(1, self.weights[
                            index] + random.uniform(0, maxDifference))
                        continue
                    if p == actualPrediction:
                        self.weights[index] = max(0, self.weights[
                            index] - random.uniform(0, maxDifference))
                        if self.weights[index] == 0:
                            self._trees[index] = self._build_tree()
                            self.weights[index] = 0.5

    def get_prediction(self, data):
        sorted_predictions, _ = self._get_predictions(data)
        return sorted_predictions[0][0]

    def _get_predictions(self, data):
        predictions = [t.get_prediction(data) for t in self._trees]
        counts = {p: 0 for p in set(predictions)}
        for index, p in enumerate(predictions):
            counts[p] += self.weights[index]
        return sorted(counts.items(), key=operator.itemgetter(1),
                      reverse=True), \
               predictions


class Benchmark:
    @staticmethod
    def run(function):
        results = []
        for i in range(100):
            result = function()
            results.append(result)
            if i < 10 or i % 10 == 9:
                mean = statistics.mean(results)
                print("{} {:3.2f} {:3.2f}".format(
                    1 + i, mean,
                    statistics.stdev(results, mean) if i > 1 else 0))