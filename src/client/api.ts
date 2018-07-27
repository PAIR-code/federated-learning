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
import * as socketProxy from 'socket.io-client';
// tslint:disable-next-line:no-angle-bracket-type-assertion no-any
const socketio = (<any>socketProxy).default || socketProxy;
// tslint:disable-next-line:max-line-length
import {ModelMsg, DownloadMsg, Events, FederatedModel, deserializeVar, log, SerializedVariable, serializeVars, federated} from './common';
import {Model} from '@tensorflow/tfjs';

const CONNECTION_TIMEOUT = 10 * 1000;
const UPLOAD_TIMEOUT = 5 * 1000;

type CounterObj = {
  [key: string]: number
};

export type DownloadCallback = (msg: DownloadMsg) => void;

/**
 * Federated Learning Client API library.
 *
 * Example usage with a tf.Model:
 * ```js
 * const tensorflowModel = await tf.loadModel('a-model.json');
 * const federatedModel = new FederatedTfModel(tensorflowModel);
 * const clientAPI = new ClientAPI(federatedModel);
 * await clientAPI.connect('http://server.com');
 * await clientAPI.fitAndUpload(data.X, data.y);
 * ```
 * The server->client synchronisation happens transparently whenever the server
 * broadcasts weights.
 * The client->server sync must be triggered manually with uploadVars
 */
export class ClientAPI {
  private msg: DownloadMsg;
  private model: FederatedModel;
  private socket: SocketIOClient.Socket;
  private downloadCallbacks: DownloadCallback[];
  private x: tf.Tensor;
  private y: tf.Tensor;
  private versionUpdateCounts: CounterObj;

  /**
   * Construct a client API for federated learning that will push and pull
   * `model` updates from the server.
   * @param model - model to use with federated learning
   */
  constructor(model: FederatedModel|Model) {
    this.model = federated(model);
    this.downloadCallbacks = [msg => {
      log('download', 'modelVersion:', msg.model.version);
    }];
    // TODO: set x and y to empty tf.Tensors (with correct shape)
    this.x = null;
    this.y = null;
    this.versionUpdateCounts = {};
  }

  /**
   * @return The version of the model we're currently training
   */
  public modelVersion(): string {
    return this.msg.model.version;
  }

  /**
   * Register a new callback to be invoked whenever the client downloads new
   * weights from the server.
   */
  public onDownload(callback: DownloadCallback): void {
    this.downloadCallbacks.push(callback);
  }

  /**
   * Connect to a server, synchronise the variables to their initial values
   * @param serverURL: The URL of the server
   * @return A promise that resolves when the connection has been established
   * and variables set to their inital values.
   */
  public async connect(serverURL: string): Promise<void> {
    const msg = await this.connectTo(serverURL);
    this.msg = msg;
    this.setVars(msg.model.vars);
    this.model.setHyperparams(msg.hyperparams);
    this.versionUpdateCounts[msg.model.version] = 0;
    this.downloadCallbacks.forEach(cb => cb(msg));

    this.socket.on(Events.Download, (msg: DownloadMsg) => {
      this.msg = msg;
      this.setVars(msg.model.vars);
      this.model.setHyperparams(msg.hyperparams);
      this.versionUpdateCounts[msg.model.version] = 0;
      this.downloadCallbacks.forEach(cb => cb(msg));
    });
  }

  /**
   * Disconnect from the server.
   */
  public dispose(): void {
    this.socket.disconnect();
    log('disconnected');
  }

  /**
   * Train the model on the given examples, upload new weights to the server,
   * then revert back to the original weights (so subsequent updates are
   * relative to the same model).
   *
   * TODO: consider having this method save copies of `xs` and `ys` when there
   * are too few examples, and only doing training/uploading after reaching a
   * configurable threshold (disposing of the copies afterwards).
   *
   * @param x Training inputs
   * @param y Training labels
   */
  public async federatedUpdate(x: tf.Tensor, y: tf.Tensor): Promise<void> {
    // TODO: reshape x and y if missing batch dimension (based on model shape)
    // TODO: save leftover examples if training w/ more than minimum
    let xNew, yNew;
    if (this.x) {
      xNew = tf.concat([this.x, x]);
      yNew = tf.concat([this.y, y]);
      tf.dispose([this.x, this.y]);
    } else {
      xNew = x.clone();
      yNew = y.clone();
    }

    if (xNew.shape[0] >= this.msg.hyperparams.examplesPerUpdate) {
      // save original ID (in case it changes during training/serialization)
      const modelVersion = this.modelVersion();
      // fit the model to the new data
      await this.model.fit(xNew, yNew);
      // serialize the new weights -- in the future we could add noise here
      const newVars = await serializeVars(this.model.getVars());
      // revert our model back to its original weights
      this.setVars(this.msg.model.vars);
      // upload the updates to the server
      await this.uploadVars({version: modelVersion, vars: newVars});
      // we're done with this data
      tf.dispose([xNew, yNew]);
      this.x = null;
      this.y = null;
      this.versionUpdateCounts[modelVersion] += 1;
    } else {
      this.x = xNew;
      this.y = yNew;
    }
  }

  public numUpdates(): number {
    let numTotal = 0;
    Object.keys(this.versionUpdateCounts).forEach(k => {
      numTotal += this.versionUpdateCounts[k];
    });
    return numTotal;
  }

  public numVersions(): number {
    return Object.keys(this.versionUpdateCounts).length;
  }

  public numExamples(): number {
    if (this.x) {
      return this.x.shape[0];
    } else {
      return 0;
    }
  }

  public numExamplesPerUpdate(): number {
    return this.msg.hyperparams.examplesPerUpdate;
  }

  /**
   * Upload the current values of the tracked variables to the server
   * @return A promise that resolves when the server has recieved the variables
   */
  private async uploadVars(msg: ModelMsg): Promise<{}> {
    const prom = new Promise((resolve, reject) => {
      const rejectTimer =
          setTimeout(() => reject(`uploadVars timed out`), UPLOAD_TIMEOUT);
      this.socket.emit(Events.Upload, msg, () => {
        clearTimeout(rejectTimer);
        resolve();
      });
    });
    return prom;
  }

  protected setVars(newVars: SerializedVariable[]) {
    tf.tidy(() => {
      this.model.setVars(newVars.map(v => deserializeVar(v)));
    });
  }

  private async connectTo(serverURL: string): Promise<DownloadMsg> {
    this.socket = socketio(serverURL);
    return fromEvent<DownloadMsg>(
        this.socket, Events.Download, CONNECTION_TIMEOUT);
  }
}

async function fromEvent<T>(
    emitter: SocketIOClient.Socket, eventName: string,
    timeout: number): Promise<T> {
  return new Promise((resolve, reject) => {
           const rejectTimer = setTimeout(
               () => reject(`${eventName} event timed out`), timeout);
           const listener = (evtArgs: T) => {
             emitter.removeListener(eventName, listener);
             clearTimeout(rejectTimer);

             resolve(evtArgs);
           };
           emitter.on(eventName, listener);
         }) as Promise<T>;
}
