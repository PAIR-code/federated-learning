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

import * as tf from '@tensorflow/tfjs';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';
import * as federated from 'federated-learning-server';

// tslint:disable:max-line-length
const MODEL_URL =
    'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/web_model.pb';
const WEIGHT_MANIFEST =
    'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/weights_manifest.json';

const LEARNING_RATE = 0.01;

const MODEL_INPUT_WIDTH = 224;
const NUM_LABELS = 424;
// Load the model & set it up for training
export async function setupModel(saveDir: string):
    Promise<federated.FederatedServerModel> {
  const model = await loadFrozenModel(MODEL_URL, WEIGHT_MANIFEST);
  const weights = model.weights;

  // TODO: there must be a better way
  const nonTrainables = /(batchnorm)|(reshape)/g;

  // Make weights trainable & extract them
  const vars: tf.Variable[] = [];

  for (const weightName in weights) {
    if (!weightName.match(nonTrainables)) {
      weights[weightName] = weights[weightName].map((t: tf.Tensor) => {
        if (t.dtype === 'float32') {
          const ret = tf.variable(t);
          vars.push(ret);
          return ret;
        } else {
          return t;
        }
      });
    }
  }

  const optimizer = tf.train.sgd(LEARNING_RATE);
  const predict = (x: tf.Tensor) => model.predict(x) as tf.Tensor;
  // TODO: better to not run softmax and use softmaxCrossEntropy?
  const loss = (input: tf.Tensor, label: tf.Tensor) => {
    const preds = predict(input) as tf.Tensor;
    return tf.losses.logLoss(label, preds) as tf.Scalar;
  };

  const inputShape = [MODEL_INPUT_WIDTH, MODEL_INPUT_WIDTH, 3];
  const outputShape = [NUM_LABELS];
  const varsAndLoss = new federated.FederatedServerDynamicModel(
      {vars, saveDir, predict, loss, optimizer, inputShape, outputShape});

  return varsAndLoss;
}
