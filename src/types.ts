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
import {Model, Scalar, Tensor, Variable} from '@tensorflow/tfjs';
import {LayerVariable} from '@tensorflow/tfjs-layers/dist/variables';

export type LossFun = (inputs: Tensor, labels: Tensor) => Scalar;
export type PredFun = (inputs: Tensor) => Tensor|Tensor[];
export type VarList = Array<Variable|LayerVariable>;
export type ModelDict = {
  vars: VarList,
  loss: LossFun,
  predict: PredFun,
  model?: Model
};

export interface FederatedModel {
  setup(): Promise<ModelDict>;
}

const audioTransferLearningModelURL =
    'https://storage.googleapis.com/tfjs-speech-command-model-14w/model.json';

export class AudioTransferLearningModel implements FederatedModel {
  async setup(): Promise<ModelDict> {
    // NOTE: have to temporarily pretend that this is a browser
    const isBrowser = tf.ENV.get('IS_BROWSER');
    tf.ENV.set('IS_BROWSER', true);
    const model = await tf.loadModel(audioTransferLearningModelURL);
    tf.ENV.set('IS_BROWSER', isBrowser);

    for (let i = 0; i < 9; ++i) {
      model.layers[i].trainable = false;  // freeze conv layers
    }

    model.compile({'optimizer': 'sgd', loss: 'categoricalCrossentropy'});

    const loss = (inputs: Tensor, labels: Tensor) => {
      const logits = model.predict(inputs) as Tensor;
      const losses = tf.losses.softmaxCrossEntropy(logits, labels);
      return losses.mean() as Scalar;
    };

    return {predict: model.predict, vars: model.trainableWeights, loss, model};
  }
}
