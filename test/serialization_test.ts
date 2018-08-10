/**
 * * @license
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
import {test_util} from '@tensorflow/tfjs-core';
// tslint:disable-next-line:max-line-length
import {deserializeVar, serializeVar, stackSerialized} from 'federated-learning-client';

describe('serialization', () => {
  it('converts back and forth to SerializedVar', async () => {
    const floatTensor =
        tf.tensor3d([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]]);
    const boolTensor = tf.tensor1d([true, false], 'bool');
    const intTensor = tf.tensor2d([[1, 2], [3, 4]], [2, 2], 'int32');

    const floatSerial = await serializeVar(floatTensor);
    const boolSerial = await serializeVar(boolTensor);
    const intSerial = await serializeVar(intTensor);
    const floatTensor2 = deserializeVar(floatSerial);
    const boolTensor2 = deserializeVar(boolSerial);
    const intTensor2 = deserializeVar(intSerial);
    test_util.expectArraysClose(floatTensor, floatTensor2);
    test_util.expectArraysClose(boolTensor, boolTensor2);
    test_util.expectArraysClose(intTensor, intTensor2);
  });

  it('can stack lists of serialized variables', async () => {
    const floatTensor1 = tf.tensor3d([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    const floatTensor2 = tf.tensor3d([[[0, 3], [0, 3]], [[0, 3], [0, 3]]]);
    const floatTensor3 =
        tf.tensor3d([[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]]);

    const intTensor1 = tf.tensor2d([[1, 2], [3, 4]], [2, 2], 'int32');
    const intTensor2 = tf.tensor2d([[5, 4], [3, 2]], [2, 2], 'int32');
    const intTensor3 = tf.tensor2d([[0, 0], [0, 0]], [2, 2], 'int32');

    const vars = [
      [await serializeVar(floatTensor1), await serializeVar(intTensor1)],
      [await serializeVar(floatTensor2), await serializeVar(intTensor2)],
      [await serializeVar(floatTensor3), await serializeVar(intTensor3)]
    ];

    const stack = stackSerialized(vars);

    const floatStack = deserializeVar(stack[0]);
    const intStack = deserializeVar(stack[1]);

    expect(floatStack.dtype).toBe('float32');
    expect(intStack.dtype).toBe('int32');

    test_util.expectArraysClose(floatStack.shape, [3, 2, 2, 2]);
    test_util.expectArraysClose(intStack.shape, [3, 2, 2]);

    test_util.expectArraysClose(
        floatStack.mean(0), tf.tensor3d([[[0, 1], [0, 1]], [[0, 1], [0, 1]]]));
    test_util.expectArraysClose(
        intStack.mean(0), tf.tensor2d([[2, 2], [2, 2]]));
  });
});
