
/**
 * * @license
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
import {Tensor, test_util, Variable} from '@tensorflow/tfjs';
import EncodingDown from 'encoding-down';
import * as fs from 'fs';
import * as http from 'http';
import LevelDown from 'leveldown';
import LevelUp from 'levelup';
import * as rimraf from 'rimraf';
import * as serverSocket from 'socket.io';

import {ClientAPI} from './client/comm';
import {tensorToJson} from './serialization';
import {ServerAPI} from './server/comm';
import {ModelDB} from './server/model_db';
import {FederatedModel, VarList} from './types';

const modelVersion = '1528400733553';
const PORT = 3001;
const socketURL = `http://0.0.0.0:${PORT}`;
const initWeights =
    [tf.tensor([1, 1, 1, 1], [2, 2]), tf.tensor([1, 2, 3, 4], [1, 4])];
const updateThreshold = 2;

class MockModel implements FederatedModel {
  vars: Variable[];
  constructor(vars: Variable[]) {
    this.vars = vars;
  }
  async fit(x: Tensor, y: Tensor) {}
  setVars(vars: Tensor[]) {
    for (let i = 0; i < this.vars.length; i++) {
      this.vars[i].assign(vars[i]);
    }
  }
  getVars(): VarList {
    return this.vars;
  }
}

describe('Socket API', () => {
  let dataDir: string;
  let modelDB: ModelDB;
  let serverAPI: ServerAPI;
  let clientAPI: ClientAPI;
  let clientVars: Variable[];
  let httpServer: http.Server;

  beforeEach(async () => {
    // Set up model database with our initial weights
    dataDir = fs.mkdtempSync('/tmp/modeldb_test');
    const lvl =
        LevelUp(EncodingDown(LevelDown(dataDir), {valueEncoding: 'json'}));
    const modelVars = await Promise.all(initWeights.map(tensorToJson));
    await lvl.put('currentModelVersion', modelVersion);
    await lvl.put(modelVersion, {'vars': modelVars});
    await lvl.close();

    modelDB = new ModelDB(dataDir, updateThreshold);
    await modelDB.setup();

    // Set up the server exposing our upload/download API
    httpServer = http.createServer();
    serverAPI = new ServerAPI(modelDB, serverSocket(httpServer));
    await serverAPI.setup();
    await httpServer.listen(PORT);

    // Set up the API client with zeroed out weights
    clientVars = initWeights.map(t => tf.variable(tf.zerosLike(t)));
    const model = new MockModel(clientVars);
    clientAPI = new ClientAPI(model);
    await clientAPI.connect(socketURL);
  });

  afterEach(async () => {
    rimraf.sync(dataDir);
    await httpServer.close();
  });

  it('transmits model version on startup', () => {
    expect(clientAPI.modelVersion()).toBe(modelVersion);
  });

  it('transmits model parameters on startup', () => {
    test_util.expectArraysClose(clientVars[0], initWeights[0]);
    test_util.expectArraysClose(clientVars[1], initWeights[1]);
  });

  it('transmits updates', async () => {
    let numUpdates = await modelDB.countUpdates();
    expect(numUpdates).toBe(0);

    clientVars[0].assign(tf.tensor([2, 2, 2, 2], [2, 2]));
    const dummyX = tf.tensor2d([[0], [0]]);
    const dummyY = tf.tensor1d([0]);
    await clientAPI.federatedUpdate(dummyX, dummyY);

    numUpdates = await modelDB.countUpdates();
    expect(numUpdates).toBe(1);
  });

  it('triggers a download after enough uploads', async (done) => {
    clientAPI.onDownload((msg) => {
      expect(msg.modelVersion).not.toBe(modelVersion);
      expect(msg.modelVersion).toBe(modelDB.modelVersion);
      test_util.expectArraysClose(
          clientVars[0], tf.tensor([1.25, 1.25, 1.25, 1.25], [2, 2]));
      test_util.expectArraysClose(
          clientVars[1], tf.tensor([3.25, 2.75, 2.25, 1.75], [1, 4]));
      done();
    });

    const dummyX1 = tf.tensor2d([[0]]);            // 1 example
    const dummyX3 = tf.tensor2d([[0], [0], [0]]);  // 3 examples
    const dummyY = tf.tensor1d([0]);
    clientVars[0].assign(tf.tensor([2, 2, 2, 2], [2, 2]));
    await clientAPI.federatedUpdate(dummyX1, dummyY);

    clientVars[0].assign(tf.tensor([1, 1, 1, 1], [2, 2]));
    clientVars[1].assign(tf.tensor([4, 3, 2, 1], [1, 4]));
    await clientAPI.federatedUpdate(dummyX3, dummyY);
  });
});
