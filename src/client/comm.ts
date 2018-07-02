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

import {DataMsg, DownloadMsg, Events, UploadMsg} from '../common';
// tslint:disable-next-line:max-line-length
import {deserializeVar, SerializedVariable, serializeVar, serializeVars} from '../serialization';
import {FederatedModel} from '../types';

const CONNECTION_TIMEOUT = 10 * 1000;
const UPLOAD_TIMEOUT = 5 * 1000;

/**
 * Synchronises tf.Variables between a client and a server.
 * Example usage with bare tf.Variables:
 * ```js
 * const { loss, vars } = setupModel()
 * const sync = new VariableSynchroniser(vars)
 * const clientHyperparams = await sync.initialise('http://server.com')
 * // train the model or otherwise update vars
 * const metaInfo = { nSteps: 1 } // optional
 * await sync.uploadVars(metaInfo)
 * ```
 * Example usage with a tf.Model:
 * ```js
 * const model = tf.loadModel('a-model.json');
 * const sync = VariableSynchroniser.fromLayers(model.getLayer('classifier'))
 * const clientFitConfig = await sync.initialise('http://server.com')
 * const h = await model.fit(data.X, data.y, clientFitConfig)
 * await sync.uploadVars(metaInfo)
 * ```
 * The server->client synchronisation happens transparently whenever the server
 * broadcasts weights.
 * The client->server sync must be triggered manually with uploadVars
 */

type UpdateCallback = (msg: DownloadMsg) => void;

export class ClientAPI {
  public model: FederatedModel;
  private msg: DownloadMsg;
  private socket: SocketIOClient.Socket;
  private updateCallbacks: UpdateCallback[];
  /**
   * Construct a synchroniser from a list of tf.Variables of tf.LayerVariables.
   * @param {Array<Variable|LayerVariable>} vars - Variables to track and sync
   */
  constructor(model: FederatedModel) {
    this.model = model;
    this.updateCallbacks = [];
  }

  public modelVersion() {
    return this.msg.modelId
  }

  public onUpdate(callback: UpdateCallback) {
    this.updateCallbacks.push(callback);
  }

  private async connect(url: string): Promise<DownloadMsg> {
    this.socket = socketio(url);
    return fromEvent<DownloadMsg>(
        this.socket, Events.Download, CONNECTION_TIMEOUT);
  }

  /**
   * Connect to a server, synchronise the variables to their
   * initial values and return the hyperparameters for this client
   * @param url: The URL of the server
   * @return A promise that resolves when the connection has been established
   * and variables set to their inital values.
   */
  public async connectTo(serverURL: string): Promise<void> {
    this.msg = await this.connect(serverURL);
    this.setVars(this.msg.vars);
    for (let i = 0; i < this.updateCallbacks.length; i++) {
      this.updateCallbacks[i](this.msg);
    }

    this.socket.on(Events.Download, (msg: DownloadMsg) => {
      this.msg = msg;
      this.setVars(msg.vars);
      for (let i = 0; i < this.updateCallbacks.length; i++) {
        this.updateCallbacks[i](this.msg);
      }
    });
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
    const modelId = this.msg.modelId;
    // fit the model to the new data
    await this.model.fit(xs, ys);
    // serialize the new weights -- in the future we could add noise here
    const newVars = await serializeVars(this.model.getVars());
    // revert our model back to its original weights
    this.setVars(this.msg.vars);
    // upload the updates to the server
    await this.uploadVars({modelId, numExamples: xs.shape[0], vars: newVars});
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
      this.model.setVars(newVars.map(deserializeVar));
    });
  }

  public dispose() {
    this.socket.disconnect();
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
