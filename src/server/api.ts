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
import {DEFAULT_HYPERPARAMS, deserializeVars, DownloadMsg, Events, federated, FederatedModel, HyperparamsMsg, log, ModelMsg, SerializedVariable, serializeVars, stackSerialized} from './common';

type UpdateCallback = (version: string) => void;

export class ServerAPI {
  model: FederatedModel;
  hyperparams: HyperparamsMsg;
  currentModel: DownloadMsg;
  server: io.Server;
  modelDir: string;
  modelVersion: string;
  numClients = 0;
  updating = false;
  updates: SerializedVariable[][] = [];
  updateCallbacks: UpdateCallback[] = [];
  aggregation = 'mean';

  constructor(
      model: FederatedModel|tf.Model, modelVersion: string, modelDir: string,
      server: io.Server, hyperparams?: HyperparamsMsg,
      private exitOnClientExit = false) {
    this.model = federated(model, hyperparams);
    this.modelDir = modelDir;
    this.modelVersion = modelVersion;
    this.server = server;
    this.hyperparams =
        Object.assign(Object.create(DEFAULT_HYPERPARAMS), hyperparams || {});
    this.currentModel = null;
    this.downloadMsg().then(msg => this.currentModel = msg);

    this.server.on('connection', async (socket: io.Socket) => {
      if (!this.currentModel) {
        this.currentModel = await this.downloadMsg();
      }

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

      socket.emit(Events.Download, this.currentModel);

      socket.on(Events.Upload, async (modelMsg: ModelMsg, ack) => {
        ack(true);
        if (modelMsg.version === this.modelVersion && !this.updating) {
          this.updates.push(modelMsg.vars);
          if (this.updates.length >= this.hyperparams.updatesPerVersion) {
            await this.updateModel();
            this.server.sockets.emit(Events.Download, this.currentModel);
          }
        }
      });
    });
  }

  onUpdate(callback: UpdateCallback) {
    this.updateCallbacks.push(callback);
  }

  async downloadMsg(): Promise<DownloadMsg> {
    const vars = await serializeVars(this.model.getVars());
    return {
      model: {
        vars,
        version: this.modelVersion,
      },
      hyperparams: this.hyperparams
    };
  }

  // TODO: optionally clip updates by global norm
  // TODO: implement median and trimmed mean aggregations
  // TODO: optionally skip updates if validation loss increases
  // TOOD: consider only updating once we achieve a certain number of _clients_
  async updateModel() {
    this.updating = true;
    const t1 = new Date().getTime();
    log(`starting update at ${t1}`);

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
    this.currentModel = await this.downloadMsg();
    this.updates = [];
    this.updating = false;
    const t2 = new Date().getTime();
    log(`finished update at ${t2} (took ${t2 - t1}ms)`);

    this.model.save(`file://${this.modelDir}/${this.modelVersion}`);
    const t3 = new Date().getTime();
    log(`saving took ${t3 - t2}ms`);
    this.updateCallbacks.forEach(c => c(this.modelVersion));
    const t4 = new Date().getTime();
    log(`callbacks took ${t4 - t3}ms`);
  }
}
