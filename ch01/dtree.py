# File: dtree.py
#    from chapter 1 of _Tree-based Machine Learning Algorithms_
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


def build(data, outcomeLabel):
    attrIndexes = [index for index, label in enumerate(data[0]) if
                   label != outcomeLabel]
    outcomeIndex = data[0].index(outcomeLabel)

    nodes = []
    lastNodeNumber = 0

    workQueue = [(-1, lastNodeNumber, set(i for i in range(1, len(data))))]
    while len(workQueue) > 0:
        parentNodeId, nodeId, dataRowIndexes = workQueue.pop()
        uniqueOutcomes = set(data[i][outcomeIndex] for i in dataRowIndexes)
        if len(uniqueOutcomes) == 1:
            nodes.append((nodeId, uniqueOutcomes.pop()))
            continue
        attrValueResults = []
        for attrIndex in attrIndexes:
            for rowIndex in dataRowIndexes:
                row = data[rowIndex]
                value = row[attrIndex]
                attrValueResults.append((attrIndex, value))
        potentials = [i for i in Counter(attrValueResults).most_common(1)]
        attrIndex, attrValue = potentials[0][0]
        matches = {rowIndex for rowIndex in dataRowIndexes if
                   data[rowIndex][attrIndex] == attrValue}
        nonMatches = dataRowIndexes - matches
        lastNodeNumber += 1
        matchId = lastNodeNumber
        workQueue.append((nodeId, matchId, matches))
        lastNodeNumber += 1
        nonMatchId = lastNodeNumber
        workQueue.append((nodeId, nonMatchId, nonMatches))
        nodes.append((nodeId, attrIndex, attrValue, matchId, nonMatchId))
    nodes = sorted(nodes, key=lambda n: n[0])
    return DTree(nodes, data[0])


class DTree:
    def __init__(self, nodes, attrNames):
        self._nodes = nodes
        self._attrNames = attrNames

    @staticmethod
    def _is_leaf(node):
        return len(node) == 2

    def __str__(self):
        s = ''
        for node in self._nodes:
            if self._is_leaf(node):
                s += '{}: {}\n'.format(node[0], node[1])
            else:
                nodeId, attrIndex, attrValue, nodeIdIfMatch, \
                    nodeIdIfNonMatch = node
                s += '{}: {}={}, Yes->{}, No->{}\n'.format(
                    nodeId, self._attrNames[attrIndex], attrValue,
                    nodeIdIfMatch, nodeIdIfNonMatch)
        return s

    def get_prediction(self, data):
        currentNode = self._nodes[0]
        while True:
            if self._is_leaf(currentNode):
                return currentNode[1]
            nodeId, attrIndex, attrValue, nodeIdIfMatch, \
                nodeIdIfNonMatch = currentNode
            currentNode = self._nodes[nodeIdIfMatch if
            data[attrIndex] == attrValue else nodeIdIfNonMatch]
