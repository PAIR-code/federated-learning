
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
import {ClientAPI} from 'federated-learning-client';
import {FederatedModel, VarList} from 'federated-learning-client';
import {ServerAPI} from 'federated-learning-server';
import * as fs from 'fs';
import * as http from 'http';
import * as rimraf from 'rimraf';
import * as serverSocket from 'socket.io';

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
  // tslint:disable-next-line:no-any
  save(path: any, config?: any) {
    return new Promise<void>(() => {});
  }
}

describe('Server-to-client API', () => {
  let dataDir: string;
  let serverAPI: ServerAPI;
  let clientAPI: ClientAPI;
  let clientVars: Variable[];
  let serverVars: Variable[];
  let httpServer: http.Server;

  beforeEach(async () => {
    // Set up model database with our initial weights
    dataDir = fs.mkdtempSync('/tmp/federated_test');

    clientVars = initWeights.map(t => tf.variable(tf.zerosLike(t)));
    serverVars = initWeights.map(t => tf.variable(t));
    const clientModel = new MockModel(clientVars);
    const serverModel = new MockModel(serverVars);

    // Set up the server exposing our upload/download API
    httpServer = http.createServer();
    await httpServer.listen(PORT);

    serverAPI = new ServerAPI(
        serverModel, modelVersion, dataDir, serverSocket(httpServer),
        updateThreshold);

    // Set up the API client with zeroed out weights
    clientAPI = new ClientAPI(clientModel);
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
    expect(serverAPI.updates.length).toBe(0);

    clientVars[0].assign(tf.tensor([2, 2, 2, 2], [2, 2]));
    const dummyX = tf.tensor2d([[0], [0]]);
    const dummyY = tf.tensor1d([0]);
    await clientAPI.federatedUpdate(dummyX, dummyY);

    expect(serverAPI.updates.length).toBe(1);
  });

  it('triggers a download after enough uploads', async (done) => {
    clientAPI.onDownload((msg) => {
      expect(msg.modelVersion).not.toBe(modelVersion);
      expect(msg.modelVersion).toBe(serverAPI.modelVersion);
      test_util.expectArraysClose(
          clientVars[0], tf.tensor([1.5, 1.5, 1.5, 1.5], [2, 2]));
      test_util.expectArraysClose(
          clientVars[1], tf.tensor([3.0, 3.0, 3.0, 2.5], [1, 4]));
      done();
    });

    const dummyX1 = tf.tensor2d([[0]]);            // 1 example
    const dummyX3 = tf.tensor2d([[0], [0], [0]]);  // 3 examples
    const dummyY = tf.tensor1d([0]);
    clientVars[0].assign(tf.tensor([2, 2, 2, 2], [2, 2]));
    clientVars[1].assign(tf.tensor([1, 2, 3, 4], [1, 4]));
    await clientAPI.federatedUpdate(dummyX1, dummyY);

    clientVars[0].assign(tf.tensor([1, 1, 1, 1], [2, 2]));
    clientVars[1].assign(tf.tensor([5, 4, 3, 1], [1, 4]));
    await clientAPI.federatedUpdate(dummyX3, dummyY);
  });
});
