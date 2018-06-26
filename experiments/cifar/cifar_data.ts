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

import {scalar, tensor1d, tensor4d, tidy} from '@tensorflow/tfjs';
import {close, open, read} from 'fs';
import {promisify} from 'util';

const IMG_SIZE = 32 * 32 * 3;
const EXAMPLE_SIZE = IMG_SIZE + 1;  // 3072 bytes for img + 1 byte for label

const EXAMPLES_PER_FILE = 10000;

const readAsync = promisify(read);
const openAsync = promisify(open);
const closeAsync = promisify(close);

const TRAIN_PATHS = [
  './cifar-10-batches-bin/data_batch_1.bin',
  './cifar-10-batches-bin/data_batch_2.bin',
  './cifar-10-batches-bin/data_batch_3.bin',
  './cifar-10-batches-bin/data_batch_4.bin',
  './cifar-10-batches-bin/data_batch_5.bin'
];

const TEST_PATHS = ['./cifar-10-batches-bin/test_batch.bin'];

export async function loadData(
    splitStart: number, splitEnd: number, trainOrTest: 'train'|'test') {
  const numExamples = splitEnd - splitStart;
  const fileIdx = Math.floor(splitStart / EXAMPLES_PER_FILE);
  const filename =
      trainOrTest === 'train' ? TRAIN_PATHS[fileIdx] : TEST_PATHS[fileIdx];

  splitStart -= fileIdx * EXAMPLES_PER_FILE;
  splitEnd -= fileIdx * EXAMPLES_PER_FILE;

  const splitStartBytes = splitStart * EXAMPLE_SIZE;
  const splitEndBytes = splitEnd * EXAMPLE_SIZE;
  const splitLenBytes = splitEndBytes - splitStartBytes;

  const resultBuf = Buffer.alloc(splitLenBytes);

  const fd = await openAsync(filename, 'r');

  let readAt = splitStartBytes;
  let writeAt = 0;
  let remainingBytes = splitLenBytes;
  while (remainingBytes > 0) {
    const {bytesRead} =
        await readAsync(fd, resultBuf, writeAt, remainingBytes, readAt);
    remainingBytes -= bytesRead;
    writeAt += bytesRead;
    readAt += bytesRead;
  }

  const closed = closeAsync(fd);

  const labels = new Uint8Array(numExamples);
  for (let i = 0; i < numExamples; i++) {
    labels[i] = resultBuf[i * EXAMPLE_SIZE];
  }

  const img = new Uint8Array(numExamples * 3072);

  for (let i = 0; i < numExamples; i++) {
    const readAt = i * EXAMPLE_SIZE + 1;
    const writeAt = i * IMG_SIZE;
    for (let j = 0; j < IMG_SIZE; j++) {
      img[writeAt + j] = resultBuf[readAt + j];
    }
  }

  const imgTensor = tidy(() => {
    const imgTensorNCHW = tensor4d(img, [numExamples, 3, 32, 32]);
    const normed = imgTensorNCHW.div(scalar(127.5)).sub(scalar(1.0));
    return normed.transpose([0, 2, 3, 1]);
  });

  const labelsTensor = tensor1d(labels, 'int32');

  await closed;

  return {imgs: imgTensor, labels: labelsTensor};
}
