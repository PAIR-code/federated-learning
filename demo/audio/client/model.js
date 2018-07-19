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

const audioTransferLearningModelURL =
    'https://storage.googleapis.com/tfjs-speech-command-model-14w/model.json';

export async function loadAudioTransferLearningModel() {
  const model = await tf.loadModel(audioTransferLearningModelURL);

  for (let i = 0; i < 9; ++i) {
    model.layers[i].trainable = false;  // freeze conv layers
  }

  const optimizer = tf.train.sgd(0.001);
  model.compile({'optimizer': optimizer, loss: 'categoricalCrossentropy'});

  return model;
}
