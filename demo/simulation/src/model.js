/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

const tf = require('@tensorflow/tfjs');

async function model() {
  const m = tf.sequential();
  m.add(tf.layers.dense({units: 10, inputShape: [2], activation: 'relu'}));
  m.add(tf.layers.dense({units: 10, activation: 'relu'}));
  m.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  return m;
}

module.exports = model;
