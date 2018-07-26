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

import * as tf from '@tensorflow/tfjs-core';

/** Parses an ArrayBuffer containing a npy file. Returns a tensor. */
export function parse(ab: ArrayBuffer): tf.Tensor {
  assert(ab.byteLength > 5);
  const view = new DataView(ab);
  let pos = 0;

  // First parse the magic string.
  const byte0 = view.getUint8(pos++);
  const magicStr = dataViewToAscii(new DataView(ab, pos, 5));
  pos += 5;
  if (byte0 !== 0x93 || magicStr !== 'NUMPY') {
    throw Error('Not a numpy file.');
  }

  // Parse the version
  const version = [view.getUint8(pos++), view.getUint8(pos++)].join('.');
  if (version !== '1.0') {
    throw Error('Unsupported version.');
  }

  // Parse the header length.
  const headerLen = view.getUint16(pos, true);
  pos += 2;

  // Parse the header.
  // header is almost json, so we just manipulated it until it is.
  //  {'descr': '<f8', 'fortran_order': False, 'shape': (1, 2), }
  const headerPy = dataViewToAscii(new DataView(ab, pos, headerLen));
  pos += headerLen;
  const bytesLeft = view.byteLength - pos;
  const headerJson = headerPy.replace('True', 'true')
                         .replace('False', 'false')
                         .replace(/'/g, `"`)
                         .replace(/,\s*}/, ' }')
                         .replace(/,?\)/, ']')
                         .replace('(', '[');
  const header = JSON.parse(headerJson);
  if (header.fortran_order) {
    throw Error('NPY parse error. Implement me.');
  }

  // Finally parse the actual data.
  const size = numEls(header.shape);
  if (header['descr'] === '<f8') {
    // 8 byte float. float64.
    assertEqual(bytesLeft, size * 8);
    const s = ab.slice(pos, pos + size * 8);
    const ta = new Float32Array(new Float64Array(s));
    return tf.tensor(ta, header.shape, 'float32');
  } else if (header['descr'] === '<f4') {
    // 4 byte float. float32.
    assertEqual(bytesLeft, size * 4);
    const s = ab.slice(pos, pos + size * 4);
    const ta = new Float32Array(s);
    return tf.tensor(ta, header.shape, 'float32');
  } else if (header['descr'] === '<i8') {
    // 8 byte int. int64.
    assertEqual(bytesLeft, size * 8);
    const s = ab.slice(pos, pos + size * 8);
    const ta = new Int32Array(s).filter((val, i) => i % 2 === 0);
    return tf.tensor(ta, header.shape, 'int32');
  } else if (header['descr'] === '|u1') {
    // uint8.
    assertEqual(bytesLeft, size);
    const s = ab.slice(pos, pos + size);
    const ta = new Uint8Array(s);
    return tf.tensor(ta, header.shape, 'int32');  // FIXME should be "uint8"
  } else {
    throw Error(`Unknown dtype "${header['descr']}". Implement me.`);
  }
}

function numEls(shape: number[]): number {
  if (shape.length === 0) {
    return 1;
  } else {
    return shape.reduce((a: number, b: number) => a * b);
  }
}

function assertEqual(actual: number, expected: number) {
  assert(
      actual === expected,
      `actual ${actual} not equal to expected ${expected}`,
  );
}

function assert(cond: boolean, msg?: string) {
  if (!cond) {
    throw Error(msg || 'assert failed');
  }
}

function dataViewToAscii(dv: DataView): string {
  let out = '';
  for (let i = 0; i < dv.byteLength; i++) {
    const val = dv.getUint8(i);
    if (val === 0) {
      break;
    }
    out += String.fromCharCode(val);
  }
  return out;
}
