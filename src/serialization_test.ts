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

import * as ser from './serialization';

describe('serialization', () => {
  const floatTensor =
      tf.tensor3d([[[1.1, 2.2], [3.3, 4.4]], [[5.5, 6.6], [7.7, 8.8]]]);
  const boolTensor = tf.tensor1d([true, false], 'bool');
  const intTensor = tf.tensor2d([[1, 2], [3, 4]], [2, 2], 'int32');

  it('converts back and forth to JSON', async () => {
    const floatJSON = await ser.tensorToJson(floatTensor);
    const boolJSON = await ser.tensorToJson(boolTensor);
    const intJSON = await ser.tensorToJson(intTensor);
    const floatTensor2 = ser.jsonToTensor(floatJSON);
    const boolTensor2 = ser.jsonToTensor(boolJSON);
    const intTensor2 = ser.jsonToTensor(intJSON);
    test_util.expectArraysClose(floatTensor, floatTensor2);
    test_util.expectArraysClose(boolTensor, boolTensor2);
    test_util.expectArraysClose(intTensor, intTensor2);
  });

  it('converts back and forth to SerializedVar', async () => {
    const floatSerial = await ser.serializeVar(floatTensor);
    const boolSerial = await ser.serializeVar(boolTensor);
    const intSerial = await ser.serializeVar(intTensor);
    const floatTensor2 = ser.deserializeVar(floatSerial);
    const boolTensor2 = ser.deserializeVar(boolSerial);
    const intTensor2 = ser.deserializeVar(intSerial);
    test_util.expectArraysClose(floatTensor, floatTensor2);
    test_util.expectArraysClose(boolTensor, boolTensor2);
    test_util.expectArraysClose(intTensor, intTensor2);
  });

  it('works for an arbitrary chain', async () => {
    const floatTensor2 = ser.jsonToTensor(await ser.serializedToJson(
        await ser.jsonToSerialized(await ser.tensorToJson(floatTensor))));
    test_util.expectArraysClose(floatTensor2, floatTensor);
  });
});
