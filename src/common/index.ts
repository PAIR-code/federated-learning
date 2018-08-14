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
// tslint:disable-next-line:max-line-length
import {LayerVariable, ModelCompileConfig, Tensor, Variable} from '@tensorflow/tfjs';

export type VarList = Array<Tensor|Variable|LayerVariable>;

export type SerializedVariable = {
  dtype: tf.DataType,
  shape: number[],
  data: ArrayBuffer
};

export const dtypeToTypedArrayCtor = {
  'float32': Float32Array,
  'int32': Int32Array,
  'bool': Uint8Array
};

export async function serializeVar(variable: tf.Tensor):
    Promise<SerializedVariable> {
  const data = await variable.data();
  // small TypedArrays are views into a larger buffer
  const copy =
      data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  return {dtype: variable.dtype, shape: variable.shape.slice(), data: copy};
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

export type LossOrMetricFn = (yTrue: Tensor, yPred: Tensor) => Tensor;

export type TfModelCallback = () => Promise<tf.Model>;

export type AsyncTfModel = string|tf.Model|TfModelCallback;

export type VersionCallback = (oldVersion: string, newVersion: string) => void;

export type UploadCallback = (msg: UploadMsg) => void;

export enum Events {
  Download = 'downloadVars',
  Upload = 'uploadVars',
}

export type ModelMsg = {
  version: string,
  vars: SerializedVariable[]
};

export type UploadMsg = {
  model: ModelMsg,
  clientId: string,
  metrics?: number[]
};

export type DownloadMsg = {
  model: ModelMsg,
  hyperparams: ClientHyperparams
};

export type FederatedFitConfig = {
  learningRate?: number,
  epochs?: number,
  batchSize?: number
};

export type FederatedCompileConfig = {
  loss?: string|LossOrMetricFn,
  metrics?: string[]
};

export type ClientHyperparams = {
  batchSize?: number,          // batch size (usually not relevant)
  learningRate?: number,       // step size
  epochs?: number,             // number of steps
  examplesPerUpdate?: number,  // min examples before fitting
  weightNoiseStddev?: number   // how much noise to add to weights
};

export type ServerHyperparams = {
  aggregation?: string,
  minUpdatesPerVersion?: number
};

export const DEFAULT_CLIENT_HYPERPARAMS: ClientHyperparams = {
  examplesPerUpdate: 5,
  learningRate: 0.001,
  batchSize: 32,
  epochs: 5,
  weightNoiseStddev: 0
};

export const DEFAULT_SERVER_HYPERPARAMS: ServerHyperparams = {
  aggregation: 'mean',
  minUpdatesPerVersion: 20
};

// tslint:disable-next-line:no-any
function override(defaults: any, choices: any) {
  // tslint:disable-next-line:no-any
  const result: any = {};
  for (const key in defaults) {
    result[key] = (choices || {})[key] || defaults[key];
  }
  for (const key in (choices || {})) {
    if (!(key in defaults)) {
      throw new Error(`Unrecognized key "${key}"`);
    }
  }
  return result;
}

export function clientHyperparams(hps?: ClientHyperparams): ClientHyperparams {
  try {
    return override(DEFAULT_CLIENT_HYPERPARAMS, hps);
  } catch (err) {
    throw new Error(`Error setting clientHyperparams: ${err.message}`);
  }
}

export function serverHyperparams(hps?: ServerHyperparams): ServerHyperparams {
  try {
    return override(DEFAULT_SERVER_HYPERPARAMS, hps);
  } catch (err) {
    throw new Error(`Error setting serverHyperparams: ${err.message}`);
  }
}

export async function fetchModel(asyncModel: AsyncTfModel): Promise<tf.Model> {
  if (typeof asyncModel === 'string') {
    return await tf.loadModel(asyncModel);
  } else if (typeof asyncModel === 'function') {
    return await asyncModel();
  } else {
    return asyncModel as tf.Model;
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
  fit(x: Tensor, y: Tensor, config?: FederatedFitConfig): Promise<void>;

  /**
   * Makes predictions on input data.
   *
   * @param x `tf.Tensor` of input data.
   *
   * @return model ouputs
   */
  predict(x: Tensor): Tensor;

  /**
   * Evaluates performance on data.
   *
   * @param x `tf.Tensor` of input data.
   * @param y `tf.Tensor` of target data.
   *
   * @return An array of evaluation metrics.
   */
  evaluate(x: Tensor, y: Tensor): number[];

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
  inputShape: number[];

  /**
   * Shape of model outputs (not including the batch dimension)
   */
  outputShape: number[];
}

export class FederatedTfModel implements FederatedModel {
  model: tf.Model;
  compileConfig: ModelCompileConfig;
  private _initialModel: AsyncTfModel;

  constructor(initialModel?: AsyncTfModel, config?: FederatedCompileConfig) {
    this._initialModel = initialModel;
    this.compileConfig = {
      loss: config.loss || 'categoricalCrossentropy',
      metrics: config.metrics || ['accuracy'],
      optimizer: 'sgd'
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

  async fit(x: Tensor, y: Tensor, config?: FederatedFitConfig) {
    if (config.learningRate) {
      (this.model.optimizer as tf.SGDOptimizer)
          .setLearningRate(config.learningRate);
    }
    await this.model.fit(x, y, {
      epochs: config.epochs || DEFAULT_CLIENT_HYPERPARAMS.epochs,
      batchSize: config.batchSize || DEFAULT_CLIENT_HYPERPARAMS.batchSize
    });
  }

  predict(x: Tensor) {
    return this.model.predict(x) as Tensor;
  }

  evaluate(x: Tensor, y: Tensor) {
    return tf.tidy(() => {
      const results = this.model.evaluate(x, y);
      if (results instanceof Array) {
        return results.map(r => r.dataSync()[0]);
      } else {
        return [results.dataSync()[0]];
      }
    });
  }

  getVars(): tf.Tensor[] {
    return this.model.trainableWeights.map((v) => v.read());
  }

  // TODO: throw friendly error if passed variable of wrong shape?
  setVars(vals: tf.Tensor[]) {
    for (let i = 0; i < vals.length; i++) {
      this.model.trainableWeights[i].write(vals[i]);
    }
  }

  get inputShape() {
    return this.model.inputLayers[0].batchInputShape.slice(1);
  }

  get outputShape() {
    return (this.model.outputShape as number[]).slice(1);
  }
}

export class FederatedDynamicModel implements FederatedModel {
  isFederatedClientModel = true;
  version: string;
  vars: tf.Variable[];
  predict: (inputs: tf.Tensor) => tf.Tensor;
  loss: (labels: tf.Tensor, preds: tf.Tensor) => tf.Scalar;
  optimizer: tf.SGDOptimizer;
  inputShape: number[];
  outputShape: number[];

  constructor(args: {
    vars: tf.Variable[]; predict: (inputs: tf.Tensor) => tf.Tensor;
    loss: (labels: tf.Tensor, preds: tf.Tensor) => tf.Scalar;
    inputShape: number[];
    outputShape: number[];
  }) {
    this.vars = args.vars;
    this.predict = args.predict;
    this.loss = args.loss;
    this.optimizer = tf.train.sgd(DEFAULT_CLIENT_HYPERPARAMS.learningRate);
    this.inputShape = args.inputShape;
    this.outputShape = args.outputShape;
  }

  async setup() {
    return Promise.resolve();
  }

  async fit(x: tf.Tensor, y: tf.Tensor, config?: FederatedFitConfig):
      Promise<void> {
    if (config.learningRate) {
      this.optimizer.setLearningRate(config.learningRate);
    }
    const epochs = (config && config.epochs) || 1;
    for (let i = 0; i < epochs; i++) {
      const ret = this.optimizer.minimize(() => this.loss(y, this.predict(x)));
      tf.dispose(ret);
    }
  }

  evaluate(x: tf.Tensor, y: tf.Tensor): number[] {
    return Array.prototype.slice.call(this.loss(y, this.predict(x)).dataSync());
  }

  getVars() {
    return this.vars;
  }

  setVars(vals: tf.Tensor[]) {
    this.vars.forEach((v, i) => {
      v.assign(vals[i]);
    });
  }
}
