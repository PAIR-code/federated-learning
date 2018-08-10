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
import {FederatedCompileConfig, deserializeVars, DownloadMsg, Events, SerializedVariable, serializeVars, stackSerialized, clientHyperparams, ClientHyperparams, VersionCallback, ModelMsg, AsyncTfModel} from './common';
// tslint:disable-next-line:max-line-length
import {FederatedServerModel, isFederatedServerModel, FederatedServerTfModel} from './models';

export type FederatedServerConfig = {
  clientHyperparams?: ClientHyperparams,
  updatesPerVersion?: number,
  modelDir?: string,
  modelCompileConfig?: FederatedCompileConfig,
  verbose?: boolean
};

/**
 * Federated Learning Server library.
 *
 * Example usage with a tf.Model:
 * ```js
 * const model = await tf.loadModel('file:///a/model.json');
 * const webServer = http.createServer();
 * const fedServer = new Server(webServer, model);
 * fedServer.setup().then(() => {
 *  webServer.listen(80);
 * });
 * ```
 *
 * The server aggregates model weight updates from clients and publishes new
 * versions of the model periodically to all clients.
 */
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
  verbose: boolean;

  constructor(
    webServer: http.Server | https.Server | io.Server,
    model: AsyncTfModel | FederatedServerModel,
    config: FederatedServerConfig) {
    // Setup server
    if (webServer instanceof http.Server || webServer instanceof https.Server) {
      this.server = io(webServer);
    } else {
      this.server = webServer;
    }

    // Setup model
    if (isFederatedServerModel(model)) {
      this.model = model;
    } else {
      const defaultDir = path.resolve(`${process.cwd()}/saved-models`);
      const modelDir = config.modelDir || defaultDir;
      const compileConfig = config.modelCompileConfig || {};
      this.model = new FederatedServerTfModel(modelDir, model, compileConfig);
    }
    this.verbose = (!!config.verbose) || (!!process.env.VERBOSE) || false;
    this.updatesPerVersion = config.updatesPerVersion || 10;
    this.clientHyperparams = clientHyperparams(config.clientHyperparams || {});
    this.downloadMsg = null;
    this.versionCallbacks = [
      (model, v1, v2) => {
        this.log(`updated model: ${v1} -> ${v2}`);
      }
    ];
  }

  /**
   * Set up the federated learning server.
   *
   * This mainly delegates to `FederatedServerModel.setup` but also performs
   * any user-defined callbacks and initializes the websocket server.
   */
  async setup() {
    await this.time('setting up model', async () => {
      await this.model.setup();
    });

    this.downloadMsg = await this.computeDownloadMsg();
    await this.performCallbacks();

    this.server.on('connection', async (socket: io.Socket) => {
      if (!this.downloadMsg) {
        this.downloadMsg = await this.computeDownloadMsg();
      }

      this.numClients++;
      this.log(`connection: ${this.numClients} clients`);

      socket.on('disconnect', () => {
        this.numClients--;
        this.log(`disconnection: ${this.numClients} clients`);
      });

      socket.emit(Events.Download, this.downloadMsg);

      socket.on(Events.Upload, async (msg: ModelMsg, ack) => {
        ack(true);
        if (msg.version === this.model.version && !this.updating) {
          this.log(`new update from ${socket.client.id}`);
          this.updates.push(msg.vars);
          if (this.updates.length >= this.updatesPerVersion) {
            await this.updateModel();
            this.server.sockets.emit(Events.Download, this.downloadMsg);
          }
        }
      });
    });
  }

  /**
   * Register a new callback to be invoked whenever the server creates a new
   * version of the model
   * @param callback function to be called on each version update.
   */
  onNewVersion(callback: VersionCallback) {
    this.versionCallbacks.push(callback);
  }

  private async computeDownloadMsg(): Promise<DownloadMsg> {
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
  private async updateModel() {
    this.updating = true;
    const oldVersion = this.model.version;

    await this.time('computing new weights', async () => {
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
    });

    this.model.save();
    this.downloadMsg = await this.computeDownloadMsg();
    this.updates = [];
    this.updating = false;
    this.performCallbacks(oldVersion);
  }

  private log(...args: any[]) {
    if (this.verbose) {
      console.log('Federated Server:', ...args);
    }
  }

  private async time(msg: string, action: () => Promise<void>) {
    const t1 = new Date().getTime();
    await action();
    const t2 = new Date().getTime();
    this.log(`${msg} took ${t2 - t1}ms`)
  }

  private async performCallbacks(oldVersion?: string) {
    await this.time('performing callbacks', async () => {
      this.versionCallbacks.forEach(
        c => c(this.model, oldVersion, this.model.version));
    });
  }
}
