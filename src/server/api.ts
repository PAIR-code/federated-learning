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

export class MetricsRecorder {
  metrics = {};

  addMetrics(m, v) {
  }

  getMetrics() {
  }
}

export class InMemoryUpdateStore {
  saveDir: string;
  updates = [];
  metrics = {};

  push(msg) {
    this.updates.push(msg.weights);
    this.metrics[msg.version]['clientSide'].push(msg.metrics);
  }

  weights() {
    return deserializeVars(stackSerialized(this.updates));
  }

  metrics() {
  }

  async registerNew(model) {
    const v = model.version;
    this.metrics[v] = {
      validation: null,
      clientSide: []
    };

  }


}

export class ServerAPI {
  model: FederatedServerModel;
  hyperparams: Hyperparams;
  downloadMsg: DownloadMsg;
  server: io.Server;
  numClients = 0;
  updates: SerializedVariable[][] = [];
  clientMetrics = {}
  validMetrics
  updating = false;
  aggregation = 'mean';
  updateCallbacks: UpdateCallback[] = [];
  updatesPerVersion: number;

  constructor(
    server: io.Server, 
    model: FederatedServerModel|tf.Model,
    hyperparams?: HyperparamsMsg,
    updatesPerVersion?: number,
    private exitOnClientExit = false) {

    this.server = server;
    this.model = federatedServer(model);
    this.hyperparams = federatedHyperparams(hyperparams);
    this.updatesPerVersion = updatesPerVersion || 10;
    this.downloadMsg = null;
    this.computeDownloadMsg().then(msg => this.downloadMsg = msg);

    this.server.on('connection', async (socket: io.Socket) => {
      if (!this.downloadMsg) {
        this.downloadMsg = await this.computeDownloadMsg();
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

      socket.emit(Events.Download, this.downloadMsg);

      socket.on(Events.Upload, async (msg: UploadMsg, ack) => {
        ack(true);
        if (msg.version === this.model.version && !this.updating) {
          this.updateStore.push(msg);
          //this.metrics.push(msg.metrics);
          //this.updates.push(msg.weights);
          if (this.updateStore.shouldUpdate()) {
            await this.updateModel();
            this.server.sockets.emit(Events.Download, this.downloadMsg);
          }
        }
      });
    });
  }

  onUpdate(callback: UpdateCallback) {
    this.updateCallbacks.push(callback);
  }

  async computeDownloadMsg(): Promise<DownloadMsg> {
    return {
      model: {
        vars: await serializeVars(this.model.getVars()),
        version: this.model.version,
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
    this.model.save();
    this.downloadMsg = await this.computeDownloadMsg();
    this.updates = [];
    this.updating = false;
    this.updateCallbacks.forEach(c => c(this.model));
  }
}
