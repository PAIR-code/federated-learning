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

import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import {promisify} from 'util';
import {VarList} from './common';

const readdir = promisify(fs.readdir);

type ModelCallback = () => Promise<tf.Model>;

export class ServerTfModel {
  saveDir: string;
  version: string;
  getInit: ModelCallback;
  model: tf.Model;

  constructor(saveDir: string, initialModel?: string|tf.Model|ModelCallback) {
    this.saveDir = saveDir;
    if (typeof initialModel === 'string') {
      this.getInit = async () => await tf.loadModel(initialModel);
    } else if (initialModel instanceof tf.Model) {
      this.getInit = async () => initialModel;
    } else {
      this.getInit = initialModel;
    }
  }

  async setup() {
    const last = await this.last();
    if (last) {
      await this.load(last);
    } else if (this.getInit) {
      this.model = await this.getInit();
      await this.save();
    } else {
      throw new Error('no initial model provided or found');
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
    const url = `file://${this.saveDir}/${version}`;
    this.model.save(url);
  }

  async load(version: string) {
    const url = `file://${this.saveDir}/${version}/model.json`;
    this.version = version;
    this.model = await tf.loadModel(url);
  }

  getVars(): VarList {
    return this.model.trainableWeights.map((v) => v.read());
  }

  setVars(vals: tf.Tensor[]) {
    for (let i = 0; i < vals.length; i++) {
      this.model.trainableWeights[i].write(vals[i]);
    }
  }
}
