
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
import {Client, FederatedClientModel} from 'federated-learning-client';
import {Server, FederatedServerModel, FederatedFitConfig} from 'federated-learning-server';
import * as fs from 'fs';
import * as http from 'http';
import * as rimraf from 'rimraf';

let initialVersion: string;
const PORT = 3001;
const socketURL = `http://0.0.0.0:${PORT}`;
const initWeights =
  [tf.tensor([1, 1, 1, 1], [2, 2]), tf.tensor([1, 2, 3, 4], [1, 4])];
const updateThreshold = 2;

class MockModel implements FederatedServerModel, FederatedClientModel {
  isFederatedClientModel = true;
  isFederatedServerModel = true;
  inputShape = [1];
  outputShape = [1];
  vars: Variable[];
  version: string;
  constructor(vars: Variable[]) {
    this.vars = vars;
  }
  async fit(x: Tensor, y: Tensor, config?: FederatedFitConfig) {}
  async setup() {
    await this.save();
  }
  async save() {
    this.version = new Date().getTime().toString();
  }
  setVars(vars: Tensor[]) {
    for (let i = 0; i < this.vars.length; i++) {
      this.vars[i].assign(vars[i]);
    }
  }
  getVars(): Tensor[] {
    return this.vars;
  }
  predict(x: Tensor) {
    return x;
  }
  evaluate(x: Tensor, y: Tensor) {
    return [0];
  }
}

describe('Server-to-client API', () => {
  let dataDir: string;
  let server: Server;
  let client: Client;
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

    server = new Server(httpServer, serverModel, {
      modelDir: dataDir,
      updatesPerVersion: updateThreshold,
      clientHyperparams: {
        examplesPerUpdate: 1
      },
      verbose: true
    });
    console.log(server.clientHyperparams);
    await server.setup();
    initialVersion = server.model.version;
    console.log(server.clientHyperparams);
    console.log(server.downloadMsg);

    // Set up the API client with zeroed out weights
    client = new Client(socketURL, clientModel, {
      verbose: true
    });
    await client.setup();
  });

  afterEach(async () => {
    rimraf.sync(dataDir);
    await httpServer.close();
  });

  it('transmits model version on startup', () => {
    expect(client.modelVersion()).toBe(initialVersion);
  });

  it('transmits model parameters on startup', () => {
    test_util.expectArraysClose(clientVars[0], initWeights[0]);
    test_util.expectArraysClose(clientVars[1], initWeights[1]);
  });

  it('transmits updates', async () => {
    expect(server.updates.length).toBe(0);

    clientVars[0].assign(tf.tensor([2, 2, 2, 2], [2, 2]));
    const dummyX = tf.tensor2d([[0], [0]]);
    const dummyY = tf.tensor1d([0]);
    await client.federatedUpdate(dummyX, dummyY);

    expect(server.updates.length).toBe(1);
  });

  it('triggers a download after enough uploads', async (done) => {
    client.onNewVersion((_, oldVersion, newVersion) => {
      expect(oldVersion).toBe(initialVersion);
      expect(newVersion).not.toBe(initialVersion);
      expect(newVersion).toBe(server.model.version);
      test_util.expectArraysClose(
        clientVars[0], tf.tensor([1.5, 1.5, 1.5, 1.5], [2, 2]));
      test_util.expectArraysClose(
        clientVars[1], tf.tensor([3.0, 3.0, 3.0, 2.5], [1, 4]));
      done();
    });

    const dummyX1 = tf.tensor2d([[0]]);            // 1 example
    const dummyY1 = tf.tensor2d([[0]]);
    const dummyX3 = tf.tensor2d([[0], [0], [0]]);  // 3 examples
    const dummyY3 = tf.tensor2d([[0], [0], [0]]);
    console.log(dummyX3.shape);
    clientVars[0].assign(tf.tensor([2, 2, 2, 2], [2, 2]));
    clientVars[1].assign(tf.tensor([1, 2, 3, 4], [1, 4]));
    await client.federatedUpdate(dummyX1, dummyY1);

    clientVars[0].assign(tf.tensor([1, 1, 1, 1], [2, 2]));
    clientVars[1].assign(tf.tensor([5, 4, 3, 1], [1, 4]));
    await client.federatedUpdate(dummyX3, dummyY3);
  });
});
