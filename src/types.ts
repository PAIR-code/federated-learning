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
import {Model, ModelFitConfig, Tensor, Variable} from '@tensorflow/tfjs';
import {LayerVariable} from '@tensorflow/tfjs-layers/dist/variables';

export type VarList = Array<Variable|LayerVariable>;

/**
 * Basic interface that users need to implement to perform federated learning.
 *
 * Any model that implements `fit`, `getVars` and `setVars` can be passed into
 * a `ClientAPI` and `ServerAPI` to do federated learning.
 */
export interface FederatedModel {
  /**
   * Trains the model to better predict the given targets.
   *
   * @param x `tf.Tensor` of training input data.
   * @param y `tf.Tensor` of training target data.
   *
   * @return A `Promise` resolved when training is done.
   */
  fit(x: Tensor, y: Tensor): Promise<void>;

  /**
   * Gets the model's variables.
   *
   * @return A list of `tf.Variable`s or LayerVariables representing the model's
   * trainable weights.
   */
  getVars(): VarList;

  /**
   * Sets the model's variables to given values.
   *
   * @param vals An array of `tf.Tensor`s representing updated model weights
   */
  setVars(vals: Tensor[]): void;
}

/**
 * Implementation of `FederatedModel` designed to wrap a `tf.Model`.
 */
export class FederatedTfModel implements FederatedModel {
  private model: Model;
  private config: ModelFitConfig;

  /**
   * Construct a new `FederatedModel` wrapping a `tf.Model`.
   *
   * @param model An instance of `tf.Model` that has already been `compile`d.
   * @param config Optional `tf.ModelFitConfig` for training.
   */
  constructor(model: Model, config?: ModelFitConfig) {
    this.model = model;
    this.config = config || {epochs: 10, batchSize: 32};
  }

  async fit(x: Tensor, y: Tensor): Promise<void> {
    await this.model.fit(x, y, this.config);
  }

  getVars(): VarList {
    return this.model.trainableWeights;
  }

  setVars(vals: Tensor[]) {
    for (let i = 0; i < vals.length; i++) {
      this.model.trainableWeights[i].write(vals[i]);
    }
  }
}

const audioTransferLearningModelURL =
    'https://storage.googleapis.com/tfjs-speech-command-model-14w/model.json';

export async function loadAudioTransferLearningModel(): Promise<Model> {
  // NOTE: have to temporarily pretend that this is a browser
  const isBrowser = tf.ENV.get('IS_BROWSER');
  tf.ENV.set('IS_BROWSER', true);
  const model = await tf.loadModel(audioTransferLearningModelURL);
  tf.ENV.set('IS_BROWSER', isBrowser);

  for (let i = 0; i < 9; ++i) {
    model.layers[i].trainable = false;  // freeze conv layers
  }

  model.compile({'optimizer': 'sgd', loss: 'categoricalCrossentropy'});

  return model;
}
