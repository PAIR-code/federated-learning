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
import EncodingDown from 'encoding-down';
import LevelDown from 'leveldown';
import LevelUp from 'levelup';
import {LevelUp as LevelDB} from 'levelup';
import * as uuid from 'uuid/v4';

// tslint:disable-next-line:max-line-length
import {DataJson, FederatedModel, jsonToTensor, ModelJson, TensorJson, tensorToJson, UpdateJson} from './common';

const DEFAULT_MIN_UPDATES = 5;

function currentTimestamp() {
  return new Date().getTime().toString();
}

export class ModelDB {
  dataDir: string;
  modelVersion: string;
  updating: boolean;
  minUpdates: number;
  db: LevelDB;

  constructor(dataDir: string, minUpdates?: number) {
    this.dataDir = dataDir;
    this.updating = false;
    this.minUpdates = minUpdates || DEFAULT_MIN_UPDATES;
    this.modelVersion = null;
  }

  async setup(model?: FederatedModel) {
    this.db = await LevelUp(
        EncodingDown(LevelDown(this.dataDir), {valueEncoding: 'json'}));
    try {
      this.modelVersion = await this.db.get('currentModelVersion');
    } catch {
      await this.writeNewVars(model.getVars() as tf.Tensor[]);
    }
  }

  async putData(data: DataJson): Promise<void> {
    return this.db.put('data/' + currentTimestamp() + '_' + uuid(), data);
  }

  async getData(): Promise<DataJson[]> {
    return new Promise((resolve, reject) => {
             const data: DataJson[] = [];
             this.db.createValueStream({gt: 'data/', lt: 'data/z'})
                 .on('data', (datum: DataJson) => data.push(datum))
                 .on('error', (error) => reject(error))
                 .on('end', () => resolve(data));
           }) as Promise<DataJson[]>;
  }

  async putUpdate(update: UpdateJson): Promise<void> {
    return this.db.put(update.modelVersion + '/' + uuid(), update);
  }

  async getUpdates(): Promise<UpdateJson[]> {
    const min = this.modelVersion;
    const max = (parseInt(min, 10) + 1).toString();
    return new Promise((resolve, reject) => {
             const updates: UpdateJson[] = [];
             this.db.createValueStream({gt: min, lt: max})
                 .on('data', (data: UpdateJson) => updates.push(data))
                 .on('error', (error) => reject(error))
                 .on('end', () => resolve(updates));
           }) as Promise<UpdateJson[]>;
  }

  async countUpdates(): Promise<number> {
    const min = this.modelVersion;
    const max = (parseInt(min, 10) + 1).toString();
    return new Promise((resolve, reject) => {
             let numUpdates = 0;
             this.db.createKeyStream({gt: min, lt: max})
                 .on('data', (key) => numUpdates++)
                 .on('error', (error) => reject(error))
                 .on('end', () => resolve(numUpdates));
           }) as Promise<number>;
  }

  async getModelVars(modelVersion: string): Promise<tf.Tensor[]> {
    const model: ModelJson = await this.db.get(modelVersion);
    return model.vars.map(jsonToTensor);
  }

  async currentVars(): Promise<tf.Tensor[]> {
    return this.getModelVars(this.modelVersion);
  }

  async possiblyUpdate(): Promise<boolean> {
    const numUpdates = await this.countUpdates();
    if (numUpdates < this.minUpdates || this.updating) {
      return false;
    }
    // this.updating = true;
    await this.update();
    // this.updating = false;
    return true;
  }

  async update() {
    const currentVars = await this.currentVars();
    const updatedVars = currentVars.map(v => tf.zerosLike(v));
    const updatesJSON = await this.getUpdates();

    // Compute total number of examples for normalization
    let totalNumExamples = 0;
    updatesJSON.forEach((obj) => {
      totalNumExamples += obj.numExamples;
    });
    const n = tf.scalar(totalNumExamples);

    tf.tidy(() => {
      // Apply normalized updates
      updatesJSON.forEach((u) => {
        const nk = tf.scalar(u.numExamples);
        const frac = nk.div(n);
        u.vars.forEach((v: TensorJson, i: number) => {
          const update = tf.mul(frac, jsonToTensor(v));
          const oldVar = updatedVars[i];
          updatedVars[i] = oldVar.add(update);
          oldVar.dispose();
        });
      });
      return updatedVars;  // so the tidy doesn't dispose of it
    });

    n.dispose();

    // Save results and update key
    await this.writeNewVars(updatedVars);

    updatedVars.map(v => v.dispose());
    currentVars.map(v => v.dispose());
  }

  async writeNewVars(newVars: tf.Tensor[]) {
    const newModelVersion = currentTimestamp();
    const newVarsJson = await Promise.all(newVars.map(tensorToJson));
    await this.db.put(newModelVersion, {'vars': newVarsJson});
    await this.db.put('currentModelVersion', newModelVersion);
    this.modelVersion = newModelVersion;
  }
}
