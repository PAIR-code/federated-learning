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

import './fetch_polyfill';

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import {promisify} from 'util';

import {FederatedServerModel} from './abstract_server';
// tslint:disable-next-line:max-line-length
import {AsyncTfModel, dtypeToTypedArrayCtor, FederatedCompileConfig, FederatedDynamicModel, FederatedTfModel} from './common';

const readdir = promisify(fs.readdir);
const exists = promisify(fs.exists);
const mkdir = promisify(fs.mkdir);
const writeFile = promisify(fs.writeFile);
const readFile = promisify(fs.readFile);
const symlink = promisify(fs.symlink);
const unlink = promisify(fs.unlink);
const readlink = promisify(fs.readlink);

async function forceSymlink(src: string, dest: string) {
  try {
    await symlink(src, dest, 'dir');
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'EEXIST') {
      throw err;
    }
    const existingLink = await readlink(dest);
    if (src !== existingLink) {
      await unlink(dest);
      await symlink(src, dest, 'dir');
    }
  }
}

/**
 * Specific version of FederatedServerModel that wraps a `tf.Model`,
 * an async function returning a `tf.Model`, or a string that can be passed to
 * `tf.loadModel`.
 *
 * Stores models as subdirectories of `saveDir`. Different model versions are
 * identified by timestamps.
 */
export class FederatedServerTfModel extends FederatedTfModel implements
    FederatedServerModel {
  isFederatedServerModel = true;
  saveDir: string;
  version: string;

  constructor(
      saveDir: string, initialModel?: AsyncTfModel,
      config?: FederatedCompileConfig) {
    super(initialModel, config);
    this.saveDir = saveDir;
  }

  async setup() {
    if (!(await exists(this.saveDir))) {
      await mkdir(this.saveDir);
    }
    const last = await this.last();
    if (last) {
      await this.load(last);
    } else {
      tf.ENV.set('IS_BROWSER', true);  // TODO: remove me in tfjs 0.12.5
      await this.fetchInitial();
      tf.ENV.set('IS_BROWSER', false);
      await this.save();
    }
  }

  async list() {
    const models = await readdir(this.saveDir);
    const idx = models.indexOf('current');
    if (idx >= 0) {
      models.splice(idx);
    }
    models.sort();
    return models;
  }

  async last() {
    const models = await this.list();
    if (models.length) {
      return models[models.length - 1];
    } else {
      return null;
    }
  }

  async save() {
    const version = new Date().getTime().toString();
    this.version = version;
    const url = `file://${this.saveDir}/${version}`;
    await this.model.save(url);
    await forceSymlink(`${this.saveDir}/${version}`, `${this.saveDir}/current`);
  }

  async load(version: string) {
    const url = `file://${this.saveDir}/${version}/model.json`;
    this.version = version;
    this.model = await tf.loadModel(url);
    this.model.compile(this.compileConfig);
    await forceSymlink(`${this.saveDir}/${version}`, `${this.saveDir}/current`);
  }
}

export class FederatedServerDynamicModel extends FederatedDynamicModel
    implements FederatedServerModel {
  saveDir: string;
  version = '';
  isFederatedServerModel = true;

  constructor(args: {
    saveDir: string, vars: tf.Variable[];
    predict: (inputs: tf.Tensor) => tf.Tensor;
    loss: (labels: tf.Tensor, preds: tf.Tensor) => tf.Scalar;
    optimizer: tf.Optimizer;
    inputShape: number[];
    outputShape: number[];
  }) {
    super(args);
    this.saveDir = args.saveDir;
    this.save();
  }

  async setup() {
    if (!(await exists(this.saveDir))) {
      await mkdir(this.saveDir);
    }
    const last = await this.last();
    if (last) {
      await this.load(last);
    } else {
      await this.save();
    }
  }

  async list() {
    const models = await readdir(this.saveDir);
    models.sort();
    return models;
  }

  async last() {
    const models = await this.list();
    if (models.length) {
      return models[models.length - 1];
    } else {
      return null;
    }
  }

  async save() {
    const version = new Date().getTime().toString();
    this.version = version;
    const path = `${this.saveDir}/${version}/`;
    await mkdir(path);
    const jsonPath = `${path}/meta.json`;
    const binPath = `${path}/data.bin`;
    const {data, json} = await flatSerialize(this.vars);
    await writeFile(jsonPath, JSON.stringify(json));
    await writeFile(binPath, data);
  }

  async load(version: string) {
    const path = `${this.saveDir}/${version}/`;
    const jsonPath = `${path}/meta.json`;
    const binPath = `${path}/data.bin`;
    const json = JSON.parse(await readFile(jsonPath, {encoding: 'utf8'}));
    const data = await readFile(binPath);
    return flatDeserialize({data, json});
  }
}

export type FlatVars = {
  data: Uint8Array,
  json: {
    meta: Array<{shape: number[], dtype: 'float32' | 'int32' | 'bool'}>,
    byteOffsets: number[]
  }
};

function unview(a: ArrayBuffer|ArrayBufferView) {
  if (ArrayBuffer.isView(a)) {
    return a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength);
  } else {
    return a;
  }
}

export async function flatSerialize(tensors: tf.Tensor[]): Promise<FlatVars> {
  const meta = tensors.map(({shape, dtype}) => ({shape, dtype}));

  const datas = await Promise.all(tensors.map(t => t.data().then(unview)));

  const totBytes =
      datas.map(({byteLength}) => byteLength).reduce((x, y) => x + y);

  const dataArr = new Uint8Array(totBytes);

  let cursor = 0;
  const byteOffsets = [];

  for (const buf of datas) {
    dataArr.set(new Uint8Array(buf), cursor);
    byteOffsets.push(cursor);
    cursor += buf.byteLength;
  }

  return {data: dataArr, json: {meta, byteOffsets}};
}

export function flatDeserialize({data, json: {meta, byteOffsets}}: FlatVars) {
  const numels = meta.map(({shape}) => shape.reduce((x, y) => x * y, 1));

  const tensors = meta.map(({shape, dtype}, i) => {
    const ctor = dtypeToTypedArrayCtor[dtype];
    const arr = new ctor(data.buffer, byteOffsets[i], numels[i]);
    return tf.tensor(arr, shape, dtype);
  });

  return tensors;
}
