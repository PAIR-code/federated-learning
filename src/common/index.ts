import * as tf from '@tensorflow/tfjs';
import {Model, ModelFitConfig, Tensor, Variable} from '@tensorflow/tfjs';
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

export function federated(model: FederatedModel|Model): FederatedModel {
  if (model instanceof Model) {
    return new FederatedTfModel(model);
  } else {
    return model;
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
