# File: test.py
#    from chapter 3 of _Tree-based Machine Learning Algorithms_
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

data = [['Name', 'Gender', 'Marital Status', 'Age', 'Born'],
        ['William', 'male', 'Married', 37, 'Germany'],
        ['Louise', 'female', 'Single', 18, 'Germany'],
        ['Minnie', 'female', 'Single', 16, 'Texas'],
        ['Emma', 'female', 'Single', 14, 'Texas'],
        ['Henry', 'male', 'Married', 47, 'Germany'],
        ['Theo', 'male', 'Single', 17, 'Texas'],
        ]

outcomeLabel = 'Born'

tree = dtree.build(data, outcomeLabel)
print(tree)

testData = ['Sophie', 'female', 'Single', 17]
predicted = tree.get_prediction(testData)
print("predicted: {}".format(predicted))
