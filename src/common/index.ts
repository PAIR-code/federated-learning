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
import { Scalar, Tensor, Variable, ModelCompileConfig } from '@tensorflow/tfjs';
import { LayerVariable } from '@tensorflow/tfjs-layers/dist/variables';
import { LossOrMetricFn } from '@tensorflow/tfjs-layers/dist/types';

export type VarList = Array<Tensor | Variable | LayerVariable>;

export type TfModelCallback = () => Promise<tf.Model>;

export type AsyncTfModel = string | tf.Model | TfModelCallback;

type Tensors = Tensor | Tensor[];

export type SerializedVariable = {
  dtype: tf.DataType,
  shape: number[],
  data: ArrayBuffer
};

export type VersionCallback = (model: FederatedModel,
  oldVersion: string, newVersion: string) => void;

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
  hyperparams: ClientHyperparams
};

export type FitConfig = {
  learningRate?: number,
  epochs?: number,
  batchSize?: number
};

export type CompileConfig = {
  loss?: string | LossOrMetricFn,
  metrics?: string[]
}

export type ClientHyperparams = {
  batchSize?: number,          // client-side batch size (not very important)
  learningRate?: number,       // client-side step size (always important)
  epochs?: number,             // client-side number of steps (important)
  examplesPerUpdate?: number,  // client-side min examples (important)
  weightNoiseStddev?: number
};

export const DEFAULT_HYPERPARAMS: ClientHyperparams = {
  examplesPerUpdate: 5,
  learningRate: 0.001,
  batchSize: 32,
  epochs: 5,
  weightNoiseStddev: 0
};

export function clientHyperparams(hps?: ClientHyperparams): ClientHyperparams {
  const defaults = Object.create(DEFAULT_HYPERPARAMS);
  return Object.assign(defaults, hps || {});
}

export async function fetchModel(asyncModel: AsyncTfModel): Promise<tf.Model> {
  if (typeof asyncModel === 'string') {
    return await tf.loadModel(asyncModel);
  } else if (asyncModel instanceof tf.Model) {
    return asyncModel;
  } else {
    return await asyncModel();
  }
}

export interface FederatedModel {
  /**
   * Trains the model to better predict the given targets.
   *
   * @param x `tf.Tensor` of training input data.
   * @param y `tf.Tensor` of training target data.
   * @param config optional fit configuration.
   *
   * @return A `Promise` resolved when training is done.
   */
  fit(x: Tensors, y: Tensors, config?: FitConfig): Promise<void>;

  /**
   * Makes predictions on input data.
   *
   * @param x `tf.Tensor` of input data.
   *
   * @return A `Promise` of model ouputs
   */
  predict(x: Tensors): Promise<Tensors>;

  /**
   * Evaluates performance on data.
   *
   * @param x `tf.Tensor` of input data.
   * @param y `tf.Tensor` of target data.
   *
   * @return A `Promise` of evaluation metrics.
   */
  evaluate(x: Tensors, y: Tensors): Promise<number[]>;

  /**
   * Gets the model's variables.
   *
   * @return A list of `tf.Variable`s or LayerVariables representing the model's
   * trainable weights.
   */
  getVars(): Tensor[];

  /**
   * Sets the model's variables to given values.
   *
   * @param vals An array of `tf.Tensor`s representing updated model weights
   */
  setVars(vals: Tensor[]): void;

  /**
   * Shape of model inputs (not including the batch dimension)
   */
  inputShape(): number[];

  /**
   * Shape of model outputs (not including the batch dimension)
   */
  outputShape(): number[];
}

export class FederatedTfModel implements FederatedModel {
  optimizer: tf.SGDOptimizer;
  model: tf.Model;
  compileConfig: ModelCompileConfig;
  private _initialModel: AsyncTfModel;

  constructor(initialModel?: AsyncTfModel, compileConfig?: CompileConfig) {
    this._initialModel = initialModel;
    // we override this later
    this.optimizer = tf.train.sgd(DEFAULT_HYPERPARAMS.learningRate);
    this.compileConfig = {
      loss: compileConfig.loss || 'categoricalCrossentropy',
      metrics: compileConfig.metrics || ['accuracy'],
      optimizer: this.optimizer
    };
  }

  async fetchInitial() {
    if (this._initialModel) {
      this.model = await fetchModel(this._initialModel);
      this.model.compile(this.compileConfig);
    } else {
      throw new Error('no initial model provided!');
    }
  }

  async fit(x: Tensor, y: Tensor, config?: FitConfig) {
    if (config.learningRate) {
      this.optimizer.setLearningRate(config.learningRate);
    }
    await this.model.fit(x, y, {
      epochs: config.epochs || DEFAULT_HYPERPARAMS.epochs,
      batchSize: config.batchSize || DEFAULT_HYPERPARAMS.batchSize
    });
  }

  async predict(x: Tensor) {
    return this.model.predict(x);
  }

  async evaluate(x: Tensor, y: Tensor) {
    return tf.tidy(() => {
      const results = this.model.evaluate(x, y);
      if (results instanceof Array) {
        return results.map(r => r.dataSync()[0]);
      } else {
        return [results.dataSync()[0]]
      }
    });
  }

  getVars(): tf.Tensor[] {
    return this.model.trainableWeights.map((v) => v.read());
  }

  setVars(vals: tf.Tensor[]) {
    for (let i = 0; i < vals.length; i++) {
      this.model.trainableWeights[i].write(vals[i]);
    }
  }

  inputShape() {
    return this.model.inputLayers[0].batchInputShape.slice(1);
  }

  outputShape() {
    return (this.model.outputShape as number[]).slice(1);
  }
}

// Federated server models need to implement a few additional methods
export interface FederatedServerModel extends FederatedModel {
  isFederatedServerModel: boolean;

  version: string;

  /**
   * Initialize the model
   */
  setup(): Promise<void>;

  /**
   * Return a list of versions that can be `load`ed
   */
  list(): Promise<string[]>;

  /**
   * Return the most recent `load`able version
   */
  last(): Promise<string>;

  /**
   * Save the current model and update `version`.
   */
  save(): Promise<void>;

  /**
   * Load the specified version of the model.
   *
   * @param version identifier of the model
   */
  load(version: string): Promise<void>;
}

// Federated client models do not
export interface FederatedClientModel extends FederatedModel {
  isFederatedClientModel: boolean;

  setup(): Promise<void>;
}

export class FederatedClientTfModel extends FederatedTfModel
  implements FederatedClientModel {
  isFederatedClientModel = true;

  async setup() {
    await this.fetchInitial();
  }
}

export function isFederatedServerModel(model: any):
  model is FederatedServerModel {
  return model.isFederatedServerModel;
}

export function isFederatedClientModel(model: any):
  model is FederatedClientModel {
  return model.isFederatedClientModel;
}

export async function serializeVar(variable: tf.Tensor):
  Promise<SerializedVariable> {
  const data = await variable.data();
  // small TypedArrays are views into a larger buffer
  const copy =
    data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  return { dtype: variable.dtype, shape: variable.shape.slice(), data: copy };
}

export async function serializeVars(vars: VarList) {
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
  const { dtype, shape, data: dataBuffer } = serialized;
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
    public vars: Variable[],
    public evaluate: (xs: Tensors, ys: Tensors) => Promise<number[]>,
    public predict: (xs: Tensors) => Promise<Tensors>,
    public loss: (xs: Tensors, ys: Tensors) => Scalar,
    public inputShape: () => number[],
    public outputShape: () => number[],
    public optimizer: tf.SGDOptimizer) { }

  async fit(x: Tensors, y: Tensors, config?: FitConfig): Promise<void> {
    if (config.learningRate) {
      this.optimizer.setLearningRate(config.learningRate);
    }
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
}

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
