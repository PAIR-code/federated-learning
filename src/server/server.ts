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
import * as fs from 'fs';
import * as http from 'http';
import * as path from 'path';
import * as socketIO from 'socket.io';
import {promisify} from 'util';
import * as uuid from 'uuid/v4';

import {DownloadMsg, Events, UploadMsg} from '../common';
import {serializedToJson, serializeVar} from '../serialization';

import {ModelDB} from './model_db';

const FIT_CONFIG = {
  batchSize: 10
};

const app = express();
const server = http.createServer(app);
const io = socketIO(server);
const writeFile = promisify(fs.writeFile);
const indexPath = path.resolve(__dirname + '/../../demo/index.html');
const dataDir = path.resolve(__dirname + '/../../data');
const modelDB = new ModelDB(dataDir);

async function downloadMsg(): Promise<DownloadMsg> {
  const varsJson = await modelDB.currentVars();
  const varsSeri = await Promise.all(varsJson.map(serializeVar));
  return {fitConfig: FIT_CONFIG, modelId: modelDB.modelId, vars: varsSeri};
}

app.get('/', (req: Request, res: Response) => {
  res.sendFile(indexPath);
});

io.on('connection', async (socket: socketIO.Socket) => {
  // Send current variables to newly connected client
  const initVars = await downloadMsg();
  socket.emit(Events.Download, initVars);

  // When a client sends us updated weights
  socket.on(Events.Upload, async (msg: UploadMsg, ack) => {
    // Save them to a file
    const modelId = msg.modelId;
    const updateId = uuid();
    const updatePath = path.join(dataDir, modelId, updateId + '.json');
    const updatedVars = await Promise.all(msg.vars.map(serializedToJson));
    const updateJSON = JSON.stringify({
      clientId: socket.client.id,
      modelId: modelId,
      numExamples: msg.numExamples,
      vars: updatedVars
    });
    await writeFile(updatePath, updateJSON);

    // Let them know we're done saving
    ack(true);

    // Potentially update the model (asynchronously)
    if (modelId == modelDB.modelId) {
      const updated = await modelDB.possiblyUpdate();
      if (updated) {
        // Send new variables to all clients if we updated
        const newVars = downloadMsg();
        io.sockets.emit(Events.Download, newVars)
      }
    }
  })
});

modelDB.setup().then(() => {
  server.listen(3000, () => {
    console.log('listening on 3000');
  });
});
