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

import './fetch_polyfill';

import * as express from 'express';
import {Request, Response} from 'express';
import * as http from 'http';
import * as path from 'path';
import * as socketIO from 'socket.io';

import {FederatedModel} from '../types';

import {SocketAPI} from './comm';
import {ModelDB} from './model_db';

const app = express();
const server = http.createServer(app);
const io = socketIO(server);
const indexPath = path.resolve(__dirname + '/../../demo/index.html');
const dataDir = path.resolve(__dirname + '/../../data');
const modelDB = new ModelDB(dataDir);
const FIT_CONFIG = {
  batchSize: 10
};
const socketAPI = new SocketAPI(modelDB, FIT_CONFIG, io);

app.get('/', (req: Request, res: Response) => {
  res.sendFile(indexPath);
});

export async function setup(model: FederatedModel) {
  return modelDB.setup(model).then(() => {
    socketAPI.setup().then(() => {
      server.listen(3000, () => {
        console.log('listening on 3000');
      });
    });
  })
}
