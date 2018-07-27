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
import {Model, Optimizer, Scalar, Tensor, Variable} from '@tensorflow/tfjs';
import {LayerVariable} from '@tensorflow/tfjs-layers/dist/variables';

export type SerializedVariable = {
  dtype: tf.DataType,
  shape: number[],
  data: ArrayBuffer
};

export async function serializeVar(variable: tf.Tensor):
    Promise<SerializedVariable> {
  const data = await variable.data();
  // small TypedArrays are views into a larger buffer
  const copy =
      data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  return {dtype: variable.dtype, shape: variable.shape.slice(), data: copy};
}

export async function serializeVars(
    vars: Array<Variable|LayerVariable|Tensor>) {
  const varsP: Array<Promise<SerializedVariable>> = [];
  vars.forEach((value, key) => {
    // tslint:disable-next-line:no-any
    const lv = (value as any);
    if (lv.write != null) {
      varsP.push(serializeVar(lv.read()));
    } else {
      varsP.push(serializeVar(lv));
    }
  });
  return Promise.all(varsP);
}

export function stackSerialized(vars: SerializedVariable[][]) {
  const updateCount = vars.length;
  const weightCount = vars[0].length;
  const stackedVars = [];

  for (let wt = 0; wt < weightCount; wt++) {
    const singleVar = vars[0][wt];
    const byteLength = singleVar.data.byteLength;
    const stackedVar = new Uint8Array(byteLength * updateCount);
    for (let up = 0; up < updateCount; up++) {
      const update = vars[up][wt].data;
      // assert(update.byteLength === byteLength);
      stackedVar.set(new Uint8Array(update), up * byteLength);
    }

    stackedVars.push({
      dtype: singleVar.dtype,
      shape: [updateCount].concat(singleVar.shape),
      data: stackedVar.buffer.slice(
          stackedVar.byteOffset, stackedVar.byteOffset + stackedVar.byteLength)
    });
  }

  return stackedVars;
}

export function deserializeVars(vars: SerializedVariable[]) {
  return vars.map(deserializeVar);
}

export function serializedToArray(serialized: SerializedVariable) {
  const {dtype, shape, data: dataBuffer} = serialized;
  let data;
  // Because socket.io will deserialise JS ArrayBuffers into Nodejs Buffers
  if (dataBuffer instanceof ArrayBuffer) {
    data = dataBuffer;
    // tslint:disable-next-line no-any
  } else if ((dataBuffer as any) instanceof Buffer) {
    // tslint:disable-next-line no-any
    const dataAsBuffer = dataBuffer as any as Buffer;
    data = dataAsBuffer.buffer.slice(
        dataAsBuffer.byteOffset,
        dataAsBuffer.byteOffset + dataAsBuffer.byteLength);
  }
  const numel = shape.reduce((x, y) => x * y, 1);
  const ctor = dtypeToTypedArrayCtor[dtype];
  return new ctor(data, 0, numel);
}

export function deserializeVar(serialized: SerializedVariable): tf.Tensor {
  const array = serializedToArray(serialized);
  return tf.tensor(array, serialized.shape, serialized.dtype);
}

const dtypeToTypedArrayCtor = {
  'float32': Float32Array,
  'int32': Int32Array,
  'bool': Uint8Array
};

export type VarList = Array<Tensor|Variable|LayerVariable>;

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

  setHyperparams(hps: HyperparamsMsg): void;

  // tslint:disable-next-line:no-any
  save(handlerOrURL: any, config?: any): Promise<void>;
}

export type HyperparamsMsg = {
  batchSize?: number,          // client-side batch size (not very important)
  learningRate?: number,       // client-side step size (always important)
  epochs?: number,             // client-side number of steps (important)
  examplesPerUpdate?: number,  // client-side min examples (important)
  updatesPerVersion?: number   // server-size min updates (important)
};

export const DEFAULT_HYPERPARAMS: HyperparamsMsg = {
  examplesPerUpdate: 5,
  updatesPerVersion: 10,
  learningRate: 0.001,
  batchSize: 32,
  epochs: 5
};

/**
 * Implementation of `FederatedModel` designed to wrap a `tf.Model`.
 */
export class FederatedTfModel implements FederatedModel {
  private model: Model;
  private hyperparams: HyperparamsMsg;
  private optimizer: tf.SGDOptimizer;

  /**
   * Construct a new `FederatedModel` wrapping a `tf.Model`.
   *
   * @param model An instance of `tf.Model` that has already been `compile`d.
   * @param hyperparams Optional hyperparameters for training.
   */
  constructor(model: Model, hyperparams?: HyperparamsMsg) {
    this.model = model;
    this.hyperparams =
        Object.assign(Object.create(DEFAULT_HYPERPARAMS), hyperparams || {});
    this.optimizer = tf.train.sgd(this.hyperparams.learningRate);
    this.model.compile({
      optimizer: this.optimizer,
      loss: 'categoricalCrossentropy',  // TODO: bad assumption
      metrics: ['accuracy']             // TODO: bad assumption
    });
  }

  async fit(x: Tensor, y: Tensor): Promise<void> {
    await this.model.fit(x, y, {
      epochs: this.hyperparams.epochs,
      batchSize: this.hyperparams.batchSize
    });
  }

  setHyperparams(hps: HyperparamsMsg) {
    this.hyperparams = Object.assign(this.hyperparams, hps);
    this.optimizer.setLearningRate(this.hyperparams.learningRate);
  }

  getVars(): VarList {
    return this.model.trainableWeights.map((v) => v.read());
  }

  setVars(vals: Tensor[]) {
    for (let i = 0; i < vals.length; i++) {
      this.model.trainableWeights[i].write(vals[i]);
    }
  }

  // tslint:disable-next-line:no-any
  async save(handler: any, config?: any) {
    this.model.save(handler, config);
  }
}

/**
 * Implementation of `FederatedModel` designed to wrap a loss function and
 * variables it operates over.
 */
export class FederatedDynamicModel implements FederatedModel {
  /**
   * Construct a new `FederatedDynamicModel` wrapping loss function and
   * variables it operates over.
   *
   * @param vars Variables to upload/download from the server.
   * @param loss Loss function to pass to the optimizer.
   * @param optimizer Optimizer to optimize the model with when fit is called
   */
  constructor(
      public vars: Variable[], public loss: (xs: Tensor, ys: Tensor) => Scalar,
      public optimizer: Optimizer) {}

  async fit(x: Tensor, y: Tensor): Promise<void> {
    const lossVal = this.optimizer.minimize(() => this.loss(x, y));
    if (lossVal) {
      lossVal.dispose();
    }
  }

  getVars(): Variable[] {
    return this.vars.slice();
  }

  setVars(vals: Tensor[]): void {
    for (let i = 0; i < vals.length; i++) {
      this.vars[i].assign(vals[i]);
    }
  }

  setHyperparams(hps: HyperparamsMsg): void {}

  // tslint:disable-next-line:no-any
  async save(handler: any, config?: any) {
    return new Promise<void>(() => {
      throw new Error('not implemented');
    });
  }
}

export function federated(
    model: FederatedDynamicModel|FederatedModel|Model,
    hyperparams?: HyperparamsMsg): FederatedModel {
  if (model instanceof Model) {
    return new FederatedTfModel(model, hyperparams);
  } else {
    return model;
  }
}

export enum Events {
  Download = 'downloadVars',
  Upload = 'uploadVars',
}

export type ModelMsg = {
  version: string,
  vars: SerializedVariable[]
};

export type DownloadMsg = {
  model: ModelMsg,
  hyperparams: HyperparamsMsg
};

let LOGGING_ENABLED = (process.env != null && !!process.env.VERBOSE) || false;

export function verbose(enabled: boolean) {
  LOGGING_ENABLED = enabled;
}

// tslint:disable-next-line:no-any
export function log(...args: any[]) {
  if (LOGGING_ENABLED) {
    console.error(...args);
  }
}
