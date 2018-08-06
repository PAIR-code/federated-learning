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

import * as tf from '@tensorflow/tfjs';
import * as io from 'socket.io';
import * as http from 'http';
import * as https from 'https';
import * as path from 'path';

// tslint:disable-next-line:max-line-length
import { CompileConfig, deserializeVars, DownloadMsg, Events, SerializedVariable, serializeVars, stackSerialized, FederatedServerModel, isFederatedServerModel, clientHyperparams, ClientHyperparams, VersionCallback, ModelMsg, AsyncTfModel } from './common';
import { FederatedServerTfModel } from './models';

let LOGGING_ENABLED = (!!process.env.VERBOSE) || false;

export function verbose(enabled: boolean) {
  LOGGING_ENABLED = enabled;
}

// tslint:disable-next-line:no-any
export function log(...args: any[]) {
  if (LOGGING_ENABLED) {
    console.log('Federated Server:', ...args);
  }
}

export type FederatedServerConfig = {
  clientHyperparams?: ClientHyperparams,
  updatesPerVersion?: number,
  modelDir?: string,
  modelCompileConfig?: CompileConfig,
  verbose?: boolean
};

export class Server {
  model: FederatedServerModel;
  clientHyperparams: ClientHyperparams;
  downloadMsg: DownloadMsg;
  server: io.Server;
  numClients = 0;
  updates: SerializedVariable[][] = [];
  updating = false;
  aggregation = 'mean';
  versionCallbacks: VersionCallback[];;
  updatesPerVersion: number;

  constructor(
    server: http.Server | https.Server, model: AsyncTfModel | FederatedServerModel,
    config: FederatedServerConfig) {
    this.server = io(server);
    if (isFederatedServerModel(model)) {
      this.model = model;
    } else {
      const modelDir = config.modelDir || path.resolve(`${__dirname}/federated-server-models`);
      const compileConfig = config.modelCompileConfig || {};
      this.model = new FederatedServerTfModel(modelDir, model, compileConfig);
    }
    if (config.verbose) {
      verbose(config.verbose);
    }
    this.updatesPerVersion = config.updatesPerVersion || 10;
    this.clientHyperparams = clientHyperparams(config.clientHyperparams);
    this.downloadMsg = null;
    this.versionCallbacks = [
      (model, v1, v2) => {
        log(`updated model: ${v1} -> ${v2}`);
      }
    ];
  }

  async setup() {
    await this.model.setup();
    this.downloadMsg = await this.computeDownloadMsg();
    this.versionCallbacks.forEach(c => c(this.model, null, this.model.version));

    this.server.on('connection', async (socket: io.Socket) => {
      if (!this.downloadMsg) {
        this.downloadMsg = await this.computeDownloadMsg();
      }

      this.numClients++;
      log(`connection: ${this.numClients} clients`);

      socket.on('disconnect', () => {
        this.numClients--;
        log(`disconnection: ${this.numClients} clients`);
      });

      socket.emit(Events.Download, this.downloadMsg);

      socket.on(Events.Upload, async (msg: ModelMsg, ack) => {
        ack(true);
        if (msg.version === this.model.version && !this.updating) {
          log(`new update from ${socket.client.id}`);
          this.updates.push(msg.vars);
          if (this.updates.length >= this.updatesPerVersion) {
            await this.updateModel();
            this.server.sockets.emit(Events.Download, this.downloadMsg);
          }
        }
      });
    });
  }

  onNewVersion(callback: VersionCallback) {
    this.versionCallbacks.push(callback);
  }

  async computeDownloadMsg(): Promise<DownloadMsg> {
    return {
      model: {
        vars: await serializeVars(this.model.getVars()),
        version: this.model.version,
      },
      hyperparams: this.clientHyperparams
    };
  }

  // TODO: optionally clip updates by global norm
  // TODO: implement median and trimmed mean aggregations
  // TODO: optionally skip updates if validation loss increases
  // TOOD: consider only updating once we achieve a certain number of _clients_
  async updateModel() {
    this.updating = true;
    const oldVersion = this.model.version;

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
    const newVersion = this.model.version;
    this.downloadMsg = await this.computeDownloadMsg();
    this.updates = [];
    this.updating = false;
    this.versionCallbacks.forEach(c => c(this.model, oldVersion, newVersion));
  }
}
