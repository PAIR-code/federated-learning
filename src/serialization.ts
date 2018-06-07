import * as tf from '@tensorflow/tfjs';

export type SerializedVariable = {
  dtype: tf.DataType,
  shape: number[],
  name: string,
  data: ArrayBuffer,
  trainable: boolean
};

export async function serializeVar(variable: tf.Variable):
    Promise<SerializedVariable> {
  const data = await variable.data();
  // small TypedArrays are views into a larger buffer
  const copy =
      data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  return {
    dtype: variable.dtype,
    shape: variable.shape.slice(),
    name: variable.name,
    trainable: variable.trainable,
    data: copy
  };
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

const dtypeToTypedArrayCtor = {
  'float32': Float32Array,
  'int32': Int32Array,
  'bool': Uint8Array
};
