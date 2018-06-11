
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

// import * as tf from '@tensorflow/tfjs';
// import {test_util} from '@tensorflow/tfjs-core';
import * as tf from '@tensorflow/tfjs';
import {test_util, Variable} from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as http from 'http';
import * as path from 'path';
import * as rimraf from 'rimraf';
import * as serverSocket from 'socket.io';

import {VariableSynchroniser} from './client/comm';
import {tensorToJson} from './serialization';
import {SocketAPI} from './server/comm';
import {ModelDB} from './server/model_db';

const modelId = '1528400733553';
const batchSize = 42;
const FIT_CONFIG = {
  batchSize: batchSize
};
const PORT = 3000;
const socketURL = `http://0.0.0.0:${PORT}`;
const initWeights =
    [tf.tensor([1, 1, 1, 1], [2, 2]), tf.tensor([1, 2, 3, 4], [1, 4])];
const updateThreshold = 2;

describe('Socket API', () => {
  let dataDir: string;
  let modelDir: string;
  let modelPath: string;
  let modelDB: ModelDB;
  let serverAPI: SocketAPI;
  let clientAPI: VariableSynchroniser;
  let clientVars: Array<Variable>;
  let httpServer: http.Server;

  beforeEach(async () => {
    // Set up model database with our initial weights
    dataDir = fs.mkdtempSync('/tmp/modeldb_test');
    modelDir = path.join(dataDir, modelId);
    modelPath = path.join(dataDir, modelId + '.json');
    fs.mkdirSync(modelDir);
    const modelJSON = await Promise.all(initWeights.map(tensorToJson));
    fs.writeFileSync(modelPath, JSON.stringify({'vars': modelJSON}));
    modelDB = new ModelDB(dataDir, updateThreshold);
    await modelDB.setup();

    // Set up the server exposing our upload/download API
    httpServer = http.createServer();
    serverAPI = new SocketAPI(modelDB, FIT_CONFIG, serverSocket(httpServer));
    await serverAPI.setup();
    await httpServer.listen(PORT);

    // Set up the API client with zeroed out weights
    clientVars = initWeights.map(t => tf.variable(tf.zerosLike(t)));
    clientAPI = new VariableSynchroniser(clientVars);
    await clientAPI.initialise(socketURL);
  });

  afterEach(async () => {
    rimraf.sync(dataDir);
    await httpServer.close();
  });

  it('transmits fit config on startup', () => {
    expect(clientAPI.fitConfig.batchSize).toBe(batchSize);
  });

  it('transmits model version on startup', () => {
    expect(clientAPI.modelId).toBe(modelId);
  });

  it('transmits model parameters on startup', () => {
    test_util.expectArraysClose(clientVars[0], initWeights[0]);
    test_util.expectArraysClose(clientVars[1], initWeights[1]);
  });

  it('transmits updates', async () => {
    let updateFiles = await modelDB.listUpdateFiles();
    expect(updateFiles.length).toBe(0);

    clientVars[0].assign(tf.tensor([2, 2, 2, 2], [2, 2]));
    clientAPI.numExamples = 1;
    await clientAPI.uploadVars();

    updateFiles = await modelDB.listUpdateFiles();
    expect(updateFiles.length).toBe(1);
  });

  it('triggers a download after enough uploads', async (done) => {
    clientVars[0].assign(tf.tensor([2, 2, 2, 2], [2, 2]));
    clientAPI.numExamples = 1;
    await clientAPI.uploadVars();

    clientVars[0].assign(tf.tensor([1, 1, 1, 1], [2, 2]));
    clientVars[1].assign(tf.tensor([4, 3, 2, 1], [1, 4]));
    clientAPI.numExamples = 3;
    await clientAPI.uploadVars();

    const timeout = 100;
    let elapsed = 0;
    const interval = setInterval(() => {
      elapsed += 1;
      if (elapsed > timeout || clientAPI.modelId != modelId) {
        clearInterval(interval);
        test_util.expectArraysClose(
            clientVars[0], tf.tensor([1.25, 1.25, 1.25, 1.25], [2, 2]))
        test_util.expectArraysClose(
            clientVars[1], tf.tensor([3.25, 2.75, 2.25, 1.75], [1, 4]))
        expect(clientAPI.numExamples).toBe(0);
        expect(clientAPI.modelId).toBe(modelDB.modelId);
        done();
      }
    }, 1);
  })
});
