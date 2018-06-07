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

import * as express from 'express';
import {Request, Response} from 'express';
import * as fs from 'fs';
import * as http from 'http';
import * as path from 'path';
import * as socketIO from 'socket.io';
import * as uuid from 'uuid/v4';
import {ModelDB} from './model_db';

const app = express();
const server = http.createServer(app);
const io = socketIO(server);

const dataDir = path.resolve(__dirname + '/../data');
const indexPath = path.resolve(__dirname + '/../demo/index.html');
const modelDB = new ModelDB(dataDir);

app.get('/', (req: Request, res: Response) => {
  res.sendFile(indexPath);
});

app.post('/updates/:modelId', (req: Request, res: Response) => {
  const modelId = req.param('modelId');
  const updateId = uuid();
  const updatePath = path.join(dataDir, modelId, updateId);
  fs.writeFile(updatePath, JSON.stringify(req.params), () => {
    if (modelId === modelDB.modelId) {
      res.sendStatus(200);
      modelDB.possiblyUpdate();
    } else {
      res.sendStatus(400);
    }
  });
});

io.on('connection', (socket: socketIO.Socket) => {
  console.log('a user connected');
});

server.listen(3000, () => {
  console.log('listening on 3000');
});
