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
  // TODO(aman): is this copy neccesary? (TypedArray might be a view into a
  // shared buffer)
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

export function deserializeVar(serialized: SerializedVariable): tf.Variable {
  const {dtype, shape, name, data, trainable} = serialized;
  const ctor = dtypeToTypedArrayCtor(dtype);
  const array = new ctor(data);
  const tensor = tf.tensor(array, shape, dtype);
  return tf.variable(tensor, trainable, name, dtype);
}

function dtypeToTypedArrayCtor(dt: tf.DataType) {
  return {'float32': Float32Array, 'int32': Int32Array, 'bool': Uint8Array}[dt];
}
