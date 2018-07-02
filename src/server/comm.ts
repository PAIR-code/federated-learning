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

import {Server, Socket} from 'socket.io';

import {DataMsg, DownloadMsg, Events, UploadMsg} from '../common';
import {serializedToJson, serializeVar} from '../serialization';

import {ModelDB} from './model_db';

export class ServerAPI {
  modelDB: ModelDB;
  io: Server;
  numClients = 0;

  constructor(
      modelDB: ModelDB, io: Server,
      private exitOnClientExit = false) {
    this.modelDB = modelDB;
    this.io = io;
  }

  async downloadMsg(): Promise<DownloadMsg> {
    const varsJson = await this.modelDB.currentVars();
    const varsSeri = await Promise.all(varsJson.map(serializeVar));
    return {
      modelId: this.modelDB.modelId,
      vars: varsSeri
    };
  }

  async setup() {
    this.io.on('connection', async (socket: Socket) => {
      socket.on('disconnect', () => {
        this.numClients--;
        if (this.exitOnClientExit && this.numClients <= 0) {
          this.io.close();
          process.exit(0);
        }
      });

      this.numClients++;

      // Send current variables to newly connected client
      const initVars = await this.downloadMsg();
      socket.emit(Events.Download, initVars);

      socket.on(Events.Data, async (msg: DataMsg, ack) => {
        ack(true);
        const x = await serializedToJson(msg.x);
        const y = await serializedToJson(msg.y);
        const clientId = socket.client.id;
        await this.modelDB.putData({x, y, clientId});
      });

      // When a client sends us updated weights
      socket.on(Events.Upload, async (msg: UploadMsg, ack) => {
        // Immediately acknowledge the request
        ack(true);

        // Save weights
        const updatedVars = await Promise.all(msg.vars.map(serializedToJson));
        const update = {
          clientId: socket.client.id,
          modelId: msg.modelId,
          numExamples: msg.numExamples,
          vars: updatedVars
        };
        await this.modelDB.putUpdate(update);

        // Potentially update the model (asynchronously)
        if (msg.modelId === this.modelDB.modelId) {
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
