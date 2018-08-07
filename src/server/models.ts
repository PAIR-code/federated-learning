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
// tslint:disable-next-line:max-line-length
import {AsyncTfModel, FederatedModel, FederatedTfModel, FederatedCompileConfig} from './common';

const readdir = promisify(fs.readdir);

// Federated server models need to implement a few additional methods
export interface FederatedServerModel extends FederatedModel {
  isFederatedServerModel: boolean;
  version: string;

  /**
   * Initialize the model
   */
  setup(): Promise<void>;

  /**
   * Return a list of versions that can be `load`ed
   */
  list(): Promise<string[]>;

  /**
   * Return the most recent `load`able version
   */
  last(): Promise<string>;

  /**
   * Save the current model and update `version`.
   */
  save(): Promise<void>;

  /**
   * Load the specified version of the model.
   *
   * @param version identifier of the model
   */
  load(version: string): Promise<void>;
}

export function isFederatedServerModel(model: any):
  model is FederatedServerModel {
  return model && model.isFederatedServerModel;
}

export class FederatedServerTfModel
  extends FederatedTfModel implements FederatedServerModel {
  isFederatedServerModel = true;
  saveDir: string;
  version: string;

  constructor(saveDir: string, initialModel?: AsyncTfModel,
    config?: FederatedCompileConfig) {
    super(initialModel, config);
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
