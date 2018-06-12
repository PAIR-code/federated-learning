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

import {ModelFitConfig} from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as path from 'path';
import {Server, Socket} from 'socket.io';
import {promisify} from 'util';
import * as uuid from 'uuid/v4';

import {DownloadMsg, Events, UploadMsg} from '../common';
import {serializedToJson, serializeVar} from '../serialization';

import {ModelDB} from './model_db';

const writeFile = promisify(fs.writeFile);

export class SocketAPI {
  modelDB: ModelDB;
  fitConfig: ModelFitConfig;
  io: Server;

  constructor(modelDB: ModelDB, fitConfig: ModelFitConfig, io: Server) {
    this.modelDB = modelDB;
    this.fitConfig = fitConfig;
    this.io = io;
  }

  async downloadMsg(): Promise<DownloadMsg> {
    const varsJson = await this.modelDB.currentVars();
    const varsSeri = await Promise.all(varsJson.map(serializeVar));
    return {
      fitConfig: this.fitConfig,
      modelId: this.modelDB.modelId,
      vars: varsSeri
    };
  }

  async setup() {
    this.io.on('connection', async (socket: Socket) => {
      // Send current variables to newly connected client
      const initVars = await this.downloadMsg();
      socket.emit(Events.Download, initVars);

      // When a client sends us updated weights
      socket.on(Events.Upload, async (msg: UploadMsg, ack) => {
        // Save them to a file
        const modelId = msg.modelId;
        const updateId = uuid();
        const updatePath =
            path.join(this.modelDB.dataDir, modelId, updateId + '.json');
        const updatedVars = await Promise.all(msg.vars.map(serializedToJson));
        const updateJSON = JSON.stringify({
          clientId: socket.client.id,
          modelId,
          numExamples: msg.numExamples,
          vars: updatedVars
        });
        await writeFile(updatePath, updateJSON);

        // Let them know we're done saving
        ack(true);

        // Potentially update the model (asynchronously)
        if (modelId === this.modelDB.modelId) {
          const updated = await this.modelDB.possiblyUpdate();
          if (updated) {
            // Send new variables to all clients if we updated
            const newVars = await this.downloadMsg();
            this.io.sockets.emit(Events.Download, newVars);
          }
        }
      });
    });
  }
}
