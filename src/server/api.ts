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

import './fetch_polyfill';
import * as tf from '@tensorflow/tfjs';
import * as io from 'socket.io';

// tslint:disable-next-line:max-line-length
import {deserializeVars, DownloadMsg, Events, federated, FederatedModel, log, SerializedVariable, serializeVars, stackSerialized, UploadMsg} from './common';

export class ServerAPI {
  model: FederatedModel;
  server: io.Server;
  modelDir: string;
  modelVersion: string;
  numClients = 0;
  updating = false;
  updates: SerializedVariable[][] = [];
  aggregation = 'mean';

  constructor(
      model: FederatedModel|tf.Model, modelVersion: string, modelDir: string,
      server: io.Server, private updatesPerVersion = 10,
      private exitOnClientExit = false) {
    this.model = federated(model);
    this.modelDir = modelDir;
    this.modelVersion = modelVersion;
    this.server = server;

    this.server.on('connection', async (socket: io.Socket) => {
      this.numClients++;
      log(`connection: ${this.numClients} clients`);

      socket.on('disconnect', () => {
        this.numClients--;
        log(`disconnection: ${this.numClients} clients`);
        if (this.exitOnClientExit && this.numClients <= 0) {
          this.server.close();
          process.exit(0);
        }
      });

      socket.emit(Events.Download, await this.downloadMsg());

      socket.on(Events.Upload, async (msg: UploadMsg, ack) => {
        ack(true);
        if (msg.modelVersion === this.modelVersion && !this.updating) {
          this.updates.push(msg.vars);
          if (this.updates.length >= this.updatesPerVersion) {
            this.updateModel();
            this.server.sockets.emit(Events.Download, await this.downloadMsg());
          }
        }
      });
    });
  }

  async downloadMsg(): Promise<DownloadMsg> {
    const vars = await serializeVars(this.model.getVars());
    return {
      vars,
      modelVersion: this.modelVersion,
    };
  }

  // TODO: optionally clip updates by global norm
  // TODO: implement median and trimmed mean aggregations
  // TODO: optionally skip updates if validation loss increases
  updateModel() {
    this.updating = true;
    log(`starting update at ${new Date().getTime().toString()}`);

    const newWeights = tf.tidy(() => {
      const stacked = stackSerialized(this.updates);
      const updates = deserializeVars(stacked);
      if (this.aggregation === 'mean') {
        return updates.map(update => update.mean(0));
      } else {
        throw new Error(`unsupported aggregation ${this.aggregation}`);
      }
    });

    this.model.setVars(newWeights);
    this.modelVersion = new Date().getTime().toString();
    this.updates = [];
    this.updating = false;
    log(`finished update at ${new Date().getTime().toString()}`);

    this.model.save(`file://${this.modelDir}/${this.modelVersion}`);
  }
}
