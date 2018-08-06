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
import { promisify } from 'util';
import { AsyncTfModel, FederatedTfModel, FederatedServerModel, CompileConfig } from './common';

const readdir = promisify(fs.readdir);

export class FederatedServerTfModel extends FederatedTfModel implements FederatedServerModel {
  isFederatedServerModel = true;
  saveDir: string;
  version: string;

  constructor(saveDir: string, initialModel?: AsyncTfModel, compileConfig?: CompileConfig) {
    super(initialModel, compileConfig);
    this.saveDir = saveDir;
  }

  async setup() {
    const last = await this.last();
    if (last) {
      await this.load(last);
    } else {
      tf.ENV.set('IS_BROWSER', true);
      await this.fetchInitial();
      tf.ENV.set('IS_BROWSER', false);
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
    const url = `file://${this.saveDir}/${version}`;
    this.model.save(url);
  }

  async load(version: string) {
    const url = `file://${this.saveDir}/${version}/model.json`;
    this.version = version;
    this.model = await tf.loadModel(url);
    this.model.compile(this.compileConfig);
  }
}
