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
import {Model, ModelFitConfig, Scalar, Tensor, Variable} from '@tensorflow/tfjs';
import {Optimizer} from '@tensorflow/tfjs';
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

export function deserializeVar(serialized: SerializedVariable): tf.Tensor {
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
  const array = new ctor(data, 0, numel);
  return tf.tensor(array, shape, dtype);
}

export type TensorJson = {
  values: number[],
  shape: number[],
  dtype?: tf.DataType
};

export type ModelJson = {
  vars: TensorJson[]
};

export type UpdateJson = {
  numExamples: number,
  vars: TensorJson[],
  modelVersion?: string,
  clientId?: string
};

export type DataJson = {
  x: TensorJson,
  y: TensorJson,
  clientId?: string
};

export async function tensorToJson(t: tf.Tensor): Promise<TensorJson> {
  let data;
  // tslint:disable-next-line:no-any
  const lv = (t as any);
  if (lv.write != null) {
    data = await lv.read().data();
  } else {
    data = await t.data();
  }
  // Note: could make this async / use base64 encoding on the buffer data
  return {'values': Array.from(data), 'shape': t.shape, 'dtype': t.dtype};
}

export function jsonToTensor(j: TensorJson): tf.Tensor {
  return tf.tensor(j.values, j.shape, j.dtype || 'float32');
}

export async function serializedToJson(s: SerializedVariable):
    Promise<TensorJson> {
  return tensorToJson(deserializeVar(s));
}

export async function jsonToSerialized(j: TensorJson):
    Promise<SerializedVariable> {
  return serializeVar(jsonToTensor(j));
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
    return this.model.trainableWeights.map((v) => v.read());
  }

  setVars(vals: Tensor[]) {
    for (let i = 0; i < vals.length; i++) {
      this.model.trainableWeights[i].write(vals[i]);
    }
  }
}

/**
 * Implementation of `FederatedModel` designed to wrap a loss function and
 * variables it operates over.
 */
export class FederatedDynamicModel implements FederatedModel {
  vars: Variable[];
  config: ModelFitConfig;
  loss: (xs: Tensor, ys: Tensor) => Scalar;
  optimizer: Optimizer;
  /**
   * Construct a new `FederatedDynamicModel` wrapping loss function and
   * variables it operates over.
   *
   * @param vars Variables to upload/download from the server.
   * @param loss Loss function to pass to the optimizer.
   * @param learningRate Options ( {learningRate} ) for the optimizer
   */
  constructor(
      vars: Variable[], loss: (xs: Tensor, ys: Tensor) => Scalar,
      optimzer: Optimizer) {
    this.vars = vars;
    this.optimizer = optimzer;
    this.loss = loss;
  }

  async fit(x: Tensor, y: Tensor): Promise<void> {
    this.optimizer.minimize(() => this.loss(x, y));
  }

  getVars() {
    return this.vars.slice();
  }

  setVars(vals: Tensor[]) {
    for (let i = 0; i < vals.length; i++) {
      this.vars[i].assign(vals[i]);
    }
  }
}

export function federated(model: FederatedDynamicModel|FederatedModel|
                          Model): FederatedModel {
  if (model instanceof Model) {
    return new FederatedTfModel(model);
  } else {
    return model;
  }
}

export enum Events {
  Download = 'downloadVars',
  Upload = 'uploadVars',
  Data = 'uploadData'
}

export type UploadMsg = {
  modelVersion: string,
  vars: SerializedVariable[],
  numExamples: number
};

export type DataMsg = {
  x: SerializedVariable,
  y: SerializedVariable
};

export type DownloadMsg = {
  modelVersion: string,
  vars: SerializedVariable[]
};
