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
import * as path from 'path';
import {promisify} from 'util';

import {Model} from '../model';
import {jsonToTensor, TensorJson, tensorToJson} from '../serialization';

const DEFAULT_MIN_UPDATES = 10;
const mkdir = promisify(fs.mkdir);
const exists = promisify(fs.exists);
const readdir = promisify(fs.readdir);
const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);

async function getLatestId(dir: string) {
  const files = await readdir(dir);
  let latestId: string = null;
  files.forEach((name) => {
    if (name.endsWith('.json')) {
      const id = name.slice(0, -5);
      if (latestId == null || id > latestId) {
        latestId = id;
      }
    }
  });
  return latestId;
}

function generateNewId() {
  return new Date().getTime().toString();
}

async function readJSON(path: string) {
  const buffer = await readFile(path);
  return JSON.parse(buffer.toString());
}

export class ModelDB {
  dataDir: string;
  modelId: string;
  updating: boolean;
  minUpdates: number;

  constructor(dataDir: string, minUpdates?: number) {
    this.dataDir = dataDir;
    this.modelId = null;
    this.updating = false;
    this.minUpdates = minUpdates || DEFAULT_MIN_UPDATES;
  }

  async setup() {
    const dirExists = await exists(this.dataDir);
    if (!dirExists) {
      await mkdir(this.dataDir);
    }

    this.modelId = await getLatestId(this.dataDir);
    if (this.modelId == null) {
      const model = new Model();
      const dict = await model.setup();
      await this.writeNewVars(dict.vars as tf.Tensor[]);
    }
  }

  async listUpdateFiles(): Promise<string[]> {
    const files = await readdir(path.join(this.dataDir, this.modelId));
    return files.map((f) => {
      return path.join(this.dataDir, this.modelId, f);
    });
  }

  async currentVars(): Promise<tf.Tensor[]> {
    const file = path.join(this.dataDir, this.modelId + '.json');
    const json = await readJSON(file);
    return json['vars'].map(jsonToTensor);
  }

  async possiblyUpdate(): Promise<boolean> {
    const updateFiles = await this.listUpdateFiles();
    if (updateFiles.length < this.minUpdates || this.updating) {
      return false;
    }
    this.updating = true;
    await this.update();
    this.updating = false;
    return true;
  }

  async update() {
    const updatedVars = await this.currentVars();
    const updateFiles = await this.listUpdateFiles();
    const updatesJSON = await Promise.all(updateFiles.map(readJSON));

    // Compute total number of examples for normalization
    let totalNumExamples = 0;
    updatesJSON.forEach((obj) => {
      totalNumExamples += obj['numExamples'];
    });
    const n = tf.scalar(totalNumExamples);

    // Apply normalized updates
    updatesJSON.forEach((u) => {
      const nk = tf.scalar(u['numExamples']);
      const frac = nk.div(n);
      u['vars'].forEach((v: TensorJson, i: number) => {
        const update = jsonToTensor(v).mul(frac);
        updatedVars[i] = updatedVars[i].add(update);
      });
    });

    // Save results and update key
    await this.writeNewVars(updatedVars);
  }

  async writeNewVars(newVars: tf.Tensor[]) {
    const newModelId = generateNewId();
    const newModelDir = path.join(this.dataDir, newModelId);
    const newModelPath = path.join(this.dataDir, newModelId + '.json');
    const newModelJSON = JSON.stringify({'vars': newVars.map(tensorToJson)});
    await writeFile(newModelPath, newModelJSON);
    await mkdir(newModelDir);
    this.modelId = newModelId;
  }
}
