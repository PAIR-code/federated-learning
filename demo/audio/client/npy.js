/*!
Copyright 2018 Propel http://propel.site/.  All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// This module saves and loads from the numpy format.
// https://docs.scipy.org/doc/numpy/neps/npy-format.html

import * as tf from "@tensorflow/tfjs-core";

/** Serializes a tensor into a npy file contents. */
export function serialize(tensor) {
  const descr = new Map([["float32", "<f4"], ["int32", "<i4"]]).get(
    tensor.dtype,
  );

  // First figure out how long the file is going to be so we can create the
  // output ArrayBuffer.
  const magicStr = "NUMPY";
  const versionStr = "\x01\x00";
  const shapeStr = String(tensor.shape.join(",")) + ",";
  const [d, fo, s] = [descr, "False", shapeStr];
  let header = `{'descr': '${d}', 'fortran_order': ${fo}, 'shape': (${s}), }`;
  const unpaddedLength =
    1 + magicStr.length + versionStr.length + 2 + header.length;
  // Spaces to 16-bit align.
  const padding = " ".repeat((16 - unpaddedLength % 16) % 16);
  header += padding;
  assertEqual((unpaddedLength + padding.length) % 16, 0);
  // Either int32 or float32 for now Both 4 bytes per element.
  // TODO support uint8 and bool.
  const bytesPerElement = 4;
  const dataLen = bytesPerElement * numEls(tensor.shape);
  const totalSize = unpaddedLength + padding.length + dataLen;

  const ab = new ArrayBuffer(totalSize);
  const view = new DataView(ab);
  let pos = 0;

  // Write magic string and version.
  view.setUint8(pos++, 0x93);
  pos = writeStrToDataView(view, magicStr + versionStr, pos);

  // Write header length and header.
  view.setUint16(pos, header.length, true);
  pos += 2;
  pos = writeStrToDataView(view, header, pos);

  // Write data
  const data = tensor.dataSync();
  assertEqual(data.length, numEls(tensor.shape));
  for (let i = 0; i < data.length; i++) {
    switch (tensor.dtype) {
      case "float32":
        view.setFloat32(pos, data[i], true);
        pos += 4;
        break;

      case "int32":
        view.setInt32(pos, data[i], true);
        pos += 4;
        break;

      default:
        throw Error(`dtype ${tensor.dtype} not yet supported.`);
    }
  }
  return ab;
}

function numEls(shape) {
  if (shape.length === 0) {
    return 1;
  } else {
    return shape.reduce((a, b) => a * b);
  }
}

function writeStrToDataView(view, str, pos) {
  for (let i = 0; i < str.length; i++) {
    view.setInt8(pos + i, str.charCodeAt(i));
  }
  return pos + str.length;
}

function assertEqual(actual, expected) {
  assert(
    actual === expected,
    `actual ${actual} not equal to expected ${expected}`,
  );
}

function assert(cond, msg) {
  if (!cond) {
    throw Error(msg || "assert failed");
  }
}
