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

/** Server code */
import '@tensorflow/tfjs-node';

import '../../src/server/fetch_polyfill';

import * as express from 'express';
import * as http from 'http';
import * as path from 'path';
import * as socketIO from 'socket.io';

import {SocketAPI} from '../../src/server/comm';
import {ModelDB} from '../../src/server/model_db';
import {CifarTransferLearningModel} from './cifar_transfer_learning_model';

const app = express();
const server = http.createServer(app);
const io = socketIO(server);
const dataDir = path.resolve(process.argv[2]);
const modelDB = new ModelDB(dataDir, parseInt(process.argv[3], 10));
const FIT_CONFIG = {
  batchSize: 1
};
const socketAPI = new SocketAPI(modelDB, FIT_CONFIG, io, true);

// Plan:

async function main() {
  await modelDB.setup(new CifarTransferLearningModel());
  await socketAPI.setup();
  await server.listen(3000);

  console.log('listening on 3000');
}

main();
