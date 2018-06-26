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
// tslint:disable-next-line:max-line-length
import {scalar, tensor, Tensor, test_util, variable} from '@tensorflow/tfjs-core';
import * as fs from 'fs';
import * as path from 'path';
import * as rimraf from 'rimraf';

import {FederatedModel} from '../types';

import {ModelDB} from './model_db';

const modelId = '1528400733553';
const updateId1 = '4c382c89-30cc-4f81-9197-c26e345cfb5b';
const updateId2 = 'cdd749c0-8908-48d7-ba87-4844c831945c';

class MockModel implements FederatedModel {
  async setup() {
    const var1 = variable(tensor([0, 0, 0, 0], [2, 2]));
    const var2 = variable(tensor([-1, -1, -1, -1], [1, 4]));
    return {
      loss: (i: Tensor, l: Tensor) => scalar(1.0),
      vars: [var1, var2],
      predict: (_: Tensor) => _
    };
  }
}

describe('ModelDB', () => {
  let dataDir: string;
  let modelDir: string;
  let modelPath: string;
  let updatePath1: string;
  let updatePath2: string;

  beforeEach(() => {
    dataDir = fs.mkdtempSync('/tmp/modeldb_test');
    modelDir = path.join(dataDir, modelId);
    modelPath = path.join(dataDir, modelId + '.json');
    updatePath1 = path.join(modelDir, updateId1 + '.json');
    updatePath2 = path.join(modelDir, updateId2 + '.json');
    fs.mkdirSync(modelDir);
    fs.writeFileSync(modelPath, JSON.stringify({
      vars: [
        {values: [0, 0, 0, 0], shape: [2, 2]},
        {values: [1, 2, 3, 4], shape: [1, 4]}
      ]
    }));
    fs.writeFileSync(updatePath1, JSON.stringify({
      numExamples: 2,
      vars: [
        {values: [1, -1, 0, 0], shape: [2, 2]},
        {values: [-1, -1, -1, -1], shape: [1, 4]}
      ]
    }));
    fs.writeFileSync(updatePath2, JSON.stringify({
      numExamples: 3,
      vars: [
        {values: [0, 0, 1, -1], shape: [2, 2]},
        {values: [1, 2, 1, 2], shape: [1, 4]}
      ]
    }));
  });

  afterEach(() => {
    rimraf.sync(dataDir);
  });

  it('defaults to treating the latest model as current', async () => {
    const db = new ModelDB(dataDir);
    await db.setup(new MockModel());
    expect(db.modelId).toBe(modelId);
  });

  it('loads variables from JSON', async () => {
    const db = new ModelDB(dataDir);
    await db.setup(new MockModel());
    const vars = await db.currentVars();
    test_util.expectArraysClose(vars[0], [0, 0, 0, 0]);
    test_util.expectArraysClose(vars[1], [1, 2, 3, 4]);
    test_util.expectArraysEqual(vars[0].shape, [2, 2]);
    test_util.expectArraysEqual(vars[1].shape, [1, 4]);
    expect(vars[0].dtype).toBe('float32');
    expect(vars[1].dtype).toBe('float32');
  });

  it('updates the model using a weighted average', async () => {
    const db = new ModelDB(dataDir);
    await db.setup(new MockModel());
    await db.update();
    expect(db.modelId).not.toBe(modelId);
    const newVars = await db.currentVars();
    test_util.expectArraysClose(newVars[0], [0.4, -0.4, 0.6, -0.6]);
    test_util.expectArraysClose(newVars[1], [0.2, 0.8, 0.2, 0.8]);
  });

  it('only performs update after passing a threshold', async () => {
    const db = new ModelDB(dataDir, 3);
    await db.setup(new MockModel());
    let updated = await db.possiblyUpdate();
    expect(updated).toBe(false);
    expect(db.modelId).toBe(modelId);
    const oldUpdateFiles = await db.listUpdateFiles();
    expect(oldUpdateFiles.length).toBe(2);
    const updateId3 = 'not-necessarily-a-uuid';
    const updatePath3 = path.join(modelDir, updateId3 + '.json');
    fs.writeFileSync(updatePath3, JSON.stringify({
      numExamples: 3,
      vars: [
        {values: [0, 0, 0, 0], shape: [2, 2]},
        {values: [0, 0, 0, 0], shape: [1, 4]}
      ]
    }));
    updated = await db.possiblyUpdate();
    expect(updated).toBe(true);
    expect(db.modelId).not.toBe(modelId);
    const newUpdateFiles = await db.listUpdateFiles();
    expect(newUpdateFiles.length).toBe(0);
  });
});
