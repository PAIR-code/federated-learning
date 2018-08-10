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
const exists = promisify(fs.exists);
const mkdir = promisify(fs.mkdir);

/**
 * FederatedServerModel describes the interface that models passed to `Server`
 * must implement.
 *
 * See the FederatedModel documentation in src/common/index.ts for more details.
 */
export interface FederatedServerModel extends FederatedModel {
  isFederatedServerModel: boolean;
  version: string;

  /**
   * Initialize the model
   */
  setup(): Promise<void>;

  /**
   * Save the current model and update `version`.
   */
  save(): Promise<void>;
}

/**
 * Type guard for federated server models.
 *
 * @param model any object
 */
export function isFederatedServerModel(model: any):
  model is FederatedServerModel {
  return model && model.isFederatedServerModel;
}

/**
 * Specific version of FederatedServerModel that wraps a `tf.Model`,
 * an async function returning a `tf.Model`, or a string that can be passed to
 * `tf.loadModel`.
 *
 * Stores models as subdirectories of `saveDir`. Different model versions are
 * identified by timestamps.
 */
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
    if (!(await exists(this.saveDir))) {
      await mkdir(this.saveDir);
    }
    const last = await this.last();
    if (last) {
      await this.load(last);
    } else {
      tf.ENV.set('IS_BROWSER', true); // TODO: remove me in tfjs 0.12.5
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
