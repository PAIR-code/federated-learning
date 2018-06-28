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

export class CifarTransferLearningModel {
  async setup() {
    /* clang-format off */
    const model = tf.sequential({
      layers: [
        tf.layers.conv2d({inputShape: [32, 32, 3],
                          filters: 64,
                          kernelSize: 3,
                          activation: 'relu'}),
        tf.layers.maxPooling2d({poolSize: 2}),
        tf.layers.conv2d({filters: 64, kernelSize: 3, activation: 'relu'}),
        tf.layers.maxPooling2d({poolSize: 2}),
        tf.layers.flatten(),
        tf.layers.dense({units: 384, activation: 'relu'}),
        tf.layers.dense({units: 384, activation: 'relu'}),
        tf.layers.dense({units: 10})
      ]
    });
    /* clang-format on */

    const loss = (inputs: tf.Tensor, labels: tf.Tensor) => {
      const logits = model.predict(inputs) as tf.Tensor;
      const losses = tf.losses.softmaxCrossEntropy(labels, logits);
      return losses.mean() as tf.Scalar;
    };

    return {predict: model.predict, vars: model.trainableWeights, loss, model};
  }
}
