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
import {DataMsg, DownloadMsg, Events, UploadMsg, FederatedModel, deserializeVar, log, SerializedVariable, serializeVar, serializeVars, federated, HyperParamsMsg} from './common';
import {Model} from '@tensorflow/tfjs';

const CONNECTION_TIMEOUT = 10 * 1000;
const UPLOAD_TIMEOUT = 5 * 1000;

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
  private broadcastHyperparams: HyperParamsMsg;

  /**
   * Construct a client API for federated learning that will push and pull
   * `model` updates from the server.
   * @param model - model to use with federated learning
   */
  constructor(model: FederatedModel|Model) {
    this.model = federated(model);
    this.downloadCallbacks = [];
  }

  /**
   * @return The version of the model we're currently training
   */
  public modelVersion(): string {
    return this.msg.modelVersion;
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
    this.setVars(msg.vars);
    this.downloadCallbacks.forEach(cb => cb(msg));

    this.socket.on(Events.Download, (msg: DownloadMsg) => {
      this.msg = msg;
      this.setVars(msg.vars);
      this.downloadCallbacks.forEach(cb => cb(msg));
      log('download', 'modelVersion:', msg.modelVersion);
    });

    this.socket.on(Events.HyperParams, (msg: HyperParamsMsg) => {
      this.broadcastHyperparams = msg;
      log('hyperParams', 'hyperParams:', msg);
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
   * TODO: remove this method, move functionality to specific demos
   *
   * Upload x and y tensors to the server (for debugging/training)
   * @return A promise that resolves when the server has recieved the data
   */
  public async uploadData(x: tf.Tensor, y: tf.Tensor): Promise<{}> {
    const msg: DataMsg = {x: await serializeVar(x), y: await serializeVar(y)};
    const prom = new Promise((resolve, reject) => {
      const rejectTimer =
          setTimeout(() => reject(`uploadData timed out`), UPLOAD_TIMEOUT);

      this.socket.emit(Events.Data, msg, () => {
        clearTimeout(rejectTimer);
        resolve();
        log('uploadData');
      });
    });
    return prom;
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
   * @param xs Training inputs
   * @param ys Training labels
   */
  public async federatedUpdate(xs: tf.Tensor, ys: tf.Tensor): Promise<void> {
    // save original model ID (in case it changes during training/serialization)
    const modelVersion = this.msg.modelVersion;
    // fit the model to the new data
    await this.model.fit(xs, ys);
    // serialize the new weights -- in the future we could add noise here
    const newVars = await serializeVars(this.model.getVars());
    // revert our model back to its original weights
    this.setVars(this.msg.vars);
    // upload the updates to the server
    await this.uploadVars(
        {modelVersion, numExamples: xs.shape[0], vars: newVars});
  }

  public async hyperparams(): Promise<object> {
    if (this.broadcastHyperparams !== undefined) {
      return this.broadcastHyperparams;
    }
    return new Promise((res, rej) => {
      this.socket.emit(Events.HyperParams, (reply: HyperParamsMsg) => {
        this.broadcastHyperparams = reply;
        res(reply);
      });
    });
  }

  /**
   * Upload the current values of the tracked variables to the server
   * @return A promise that resolves when the server has recieved the variables
   */
  private async uploadVars(msg: UploadMsg): Promise<{}> {
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
