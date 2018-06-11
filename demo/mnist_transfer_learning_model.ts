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

import * as tf from '@tensorflow/tfjs'
import {Tensor} from '@tensorflow/tfjs';
import {FederatedModel, ModelDict} from '../src/types';

// https://github.com/tensorflow/tfjs-examples/tree/master/mnist-transfer-cnn
// tslint:disable-next-line:max-line-length
const mnistTransferLearningModelURL =
    'https://storage.googleapis.com/tfjs-models/tfjs/mnist_transfer_cnn_v1/model.json';

export class MnistTransferLearningModel implements FederatedModel {
  async setup(): Promise<ModelDict> {
    const model = await tf.loadModel(mnistTransferLearningModelURL);

    for (let i = 0; i < 7; ++i) {
      model.layers[i].trainable = false;  // freeze conv layers
    }

    const loss =
        (inputs: Tensor, labels: Tensor) => {
          const logits = model.predict(inputs) as Tensor;
          return tf.losses.softmaxCrossEntropy(logits, labels);
        }

    return {
      predict: model.predict, vars: model.trainableWeights, loss: loss
    }
  }
}
