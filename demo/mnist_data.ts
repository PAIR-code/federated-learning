import {tensor1d, tensor4d} from '@tensorflow/tfjs';
import {readFileSync} from 'fs';
import * as path from 'path';

const TRAIN_DATA = {
  imgs: path.resolve(__dirname, 'train-images-idx3-ubyte'),
  labels: path.resolve(__dirname, 'train-labels-idx1-ubyte')
};

function sliceIntoOwnBuffer(arr: Buffer): ArrayBuffer {
  return arr.buffer.slice(arr.byteOffset, arr.byteOffset + arr.byteLength);
}

export function loadMnist() {
  const trainImgsBytes =
      sliceIntoOwnBuffer(readFileSync(TRAIN_DATA.imgs).swap32());
  const trainLabelsBytes =
      sliceIntoOwnBuffer(readFileSync(TRAIN_DATA.labels).swap32());
  const trainImgsI32View = new Int32Array(trainImgsBytes);
  const trainLabelsI32View = new Int32Array(trainLabelsBytes);

  if (trainImgsI32View[0] !== 0x00000803) {
    throw new Error(
        'Training images file has invalid magic number 0x00000803 !== ' +
        trainImgsI32View[0].toString(16));
  }
  if (trainLabelsI32View[0] !== 0x00000801) {
    throw new Error(
        'Training labels file has invalid magic number 0x00000801 !== ' +
        trainLabelsI32View[0].toString(16));
  }

  const numItems = trainImgsI32View[1];
  const numRows = trainImgsI32View[2];
  const numCols = trainImgsI32View[3];
  const imgData =
      new Uint8Array(trainImgsBytes, 16, numItems * numRows * numCols);

  const imgsTensor =
      tensor4d(imgData, [numItems, numRows, numCols, 1], 'float32');

  if (trainLabelsI32View[1] !== numItems) {
    throw new Error(`${numItems} images but ${trainLabelsI32View[1]} labels`);
  }

  const labelsData = new Uint8Array(trainLabelsBytes, 8, numItems);
  const labelsTensor = tensor1d(labelsData, 'int32');

  return {imgs: imgsTensor, labels: labelsTensor};
}
