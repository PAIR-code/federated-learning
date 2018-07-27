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
import {log} from 'federated-learning-server';

// export const labelNames = [
//  'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
//  'zero', 'left', 'right', 'go', 'stop'
//];

export const labelNames = ['accio', 'expelliarmus', 'lumos', 'nox'];

import '@tensorflow/tfjs-node';

export async function loadAudioTransferLearningModel(url: string) {
  log(`about to load model from ${url}`);

  // NOTE: have to temporarily pretend that this is a browser
  tf.ENV.set('IS_BROWSER', true);
  const model = await tf.loadModel(url);
  tf.ENV.set('IS_BROWSER', false);

  for (let i = 0; i < 9; ++i) {
    model.layers[i].trainable = false;  // freeze conv layers
  }

  if (url.indexOf('http') >= 0) {
    const cutoffTensor = model.layers[9].output;
    const newDenseLayer1 = tf.layers.dense({units: 50, activation: 'relu'});
    const newDenseLayer2 =
        tf.layers.dense({units: labelNames.length, activation: 'softmax'});
    const newOutputTensor =
        newDenseLayer2.apply(newDenseLayer1.apply(cutoffTensor));
    const transferModel = tf.model(
        {inputs: model.inputs, outputs: newOutputTensor as tf.SymbolicTensor});
    transferModel.compile({
      loss: 'categoricalCrossentropy',
      optimizer: 'sgd',
      metrics: ['accuracy']
    });
    return transferModel;
  } else {
    model.compile({
      'optimizer': 'sgd',
      loss: 'categoricalCrossentropy',
      'metrics': ['accuracy']
    });
    return model;
  }
}
