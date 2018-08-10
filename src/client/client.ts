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
// tslint:disable-next-line:max-line-length
import {FederatedCompileConfig, VersionCallback, ModelMsg, DownloadMsg, Events, deserializeVar, SerializedVariable, serializeVars, AsyncTfModel} from './common';
// tslint:disable-next-line:max-line-length
import {FederatedClientModel, isFederatedClientModel, FederatedClientTfModel} from './models';
// tslint:disable-next-line:no-angle-bracket-type-assertion no-any
const socketio = (<any>socketProxy).default || socketProxy;
const CONNECTION_TIMEOUT = 10 * 1000;
const UPLOAD_TIMEOUT = 5 * 1000;

type CounterObj = {
  [key: string]: number
};

export type FederatedClientConfig = {
  modelCompileConfig?: FederatedCompileConfig,
  verbose?: boolean
};

/**
 * Federated Learning Client library.
 *
 * Example usage with a tf.Model:
 * ```js
 * const model = await tf.loadModel('a-model.json');
 * const client = new Client('http://server.com', model);
 * await client.setup();
 * await client.federatedUpdate(data.X, data.y);
 * ```
 * The server->client synchronisation happens transparently whenever the server
 * broadcasts weights.
 * The client->server syncs happen periodically after enough `federatedUpdate`
 * calls occur.
 */
export class Client {
  private msg: DownloadMsg;
  private model: FederatedClientModel;
  private socket: SocketIOClient.Socket;
  private versionCallbacks: VersionCallback[];
  private x: tf.Tensor;
  private y: tf.Tensor;
  private versionUpdateCounts: CounterObj;
  private serverUrl: string;
  private verbose: boolean;

  /**
   * Construct a client API for federated learning that will push and pull
   * `model` updates from the server.
   * @param model - model to use with federated learning
   */
  constructor(serverUrl: string, model: FederatedClientModel | AsyncTfModel,
    config?: FederatedClientConfig) {
    this.serverUrl = serverUrl;
    if (isFederatedClientModel(model)) {
      this.model = model;
    } else {
      const compileConfig = (config || {}).modelCompileConfig || {};
      this.model = new FederatedClientTfModel(model, compileConfig);
    }
    this.versionCallbacks = [
      (model, v1, v2) => {
        this.log(`Updated model: ${v1} -> ${v2}`);
      }
    ];
    this.versionUpdateCounts = {};
    this.verbose = (config || {}).verbose;
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
  onNewVersion(callback: VersionCallback) {
    this.versionCallbacks.push(callback);
  }

  /**
   * Connect to a server, synchronise the variables to their initial values
   * @param serverURL: The URL of the server
   * @return A promise that resolves when the connection has been established
   * and variables set to their inital values.
   */
  public async setup(): Promise<void> {
    await this.time('Initial model setup', async () => {
      await this.model.setup();
    });
    this.x = tf.tensor([], [0].concat(this.model.inputShape));
    this.y = tf.tensor([], [0].concat(this.model.outputShape));
    await this.time('Download weights from server', async () => {
      this.msg = await this.connectTo(this.serverUrl);
    });
    this.setVars(this.msg.model.vars);
    const newVersion = this.modelVersion();
    this.versionUpdateCounts[newVersion] = 0;
    this.versionCallbacks.forEach(cb => cb(this.model, null, newVersion));

    this.socket.on(Events.Download, (msg: DownloadMsg) => {
      const oldVersion = this.modelVersion();
      const newVersion = msg.model.version;
      this.msg = msg;
      this.setVars(msg.model.vars);
      this.versionUpdateCounts[newVersion] = 0;
      this.versionCallbacks.forEach(
        cb => cb(this.model, oldVersion, newVersion));
    });
  }

  /**
   * Disconnect from the server.
   */
  public dispose(): void {
    this.socket.disconnect();
    this.log('Disconnected');
  }

  /**
   * Train the model on the given examples, upload new weights to the server,
   * then revert back to the original weights (so subsequent updates are
   * relative to the same model).
   *
   * Note: this method will save copies of `x` and `y` when there
   * are too few examples and only train/upload after reaching a
   * configurable threshold (disposing of the copies afterwards).
   *
   * @param x Training inputs
   * @param y Training labels
   */
  public async federatedUpdate(x: tf.Tensor, y: tf.Tensor): Promise<void> {
    // incorporate examples into our stored `x` and `y`
    const xNew = addRows(this.x, x, this.model.inputShape);
    const yNew = addRows(this.y, y, this.model.outputShape);
    tf.dispose([this.x, this.y]);
    this.x = xNew;
    this.y = yNew;

    // repeatedly, for as many iterations as we have batches of examples:
    const examplesPerUpdate = this.msg.hyperparams.examplesPerUpdate;
    this.log(examplesPerUpdate);
    while (this.x.shape[0] >= examplesPerUpdate) {
      // save original ID (in case it changes during training/serialization)
      const modelVersion = this.modelVersion();

      // grab the right number of examples
      const xTrain = sliceWithEmptyTensors(this.x, 0, examplesPerUpdate);
      const yTrain = sliceWithEmptyTensors(this.y, 0, examplesPerUpdate);
      const fitConfig = {
        epochs: this.msg.hyperparams.epochs,
        batchSize: this.msg.hyperparams.batchSize,
        learningRate: this.msg.hyperparams.learningRate
      };

      // fit the model for the specified # of steps
      await this.time('Fit model', async () => {
        await this.model.fit(xTrain, yTrain, fitConfig);
      });

      // serialize, possibly adding noise
      const stdDev = this.msg.hyperparams.weightNoiseStddev;
      let newVars: SerializedVariable[];
      if (stdDev) {
        const newTensors = tf.tidy(() => {
          return this.model.getVars().map(v => {
            return v.add(tf.randomNormal(v.shape, 0, stdDev));
          });
        });
        newVars = await serializeVars(newTensors);
        tf.dispose(newTensors);
      } else {
        newVars = await serializeVars(this.model.getVars());
      }

      // revert our model back to its original weights
      this.setVars(this.msg.model.vars);

      // upload the updates to the server
      await this.time('Upload weights to server', async () => {
        await this.uploadVars({version: modelVersion, vars: newVars});
      });
      this.versionUpdateCounts[modelVersion] += 1;

      // dispose of the examples we saw
      // TODO: consider storing some examples longer-term and reusing them for
      // updates for multiple versions, if session is long-lived.
      tf.dispose([xTrain, yTrain]);
      const xRest = sliceWithEmptyTensors(this.x, examplesPerUpdate);
      const yRest = sliceWithEmptyTensors(this.y, examplesPerUpdate);
      tf.dispose([this.x, this.y]);
      this.x = xRest;
      this.y = yRest;
    }
  }

  public evaluate(x: tf.Tensor, y: tf.Tensor): number[] {
    return this.model.evaluate(x, y);
  }

  public predict(x: tf.Tensor): tf.Tensor {
    return this.model.predict(x);
  }

  public get inputShape(): number[] {
    return this.model.inputShape;
  }

  public get outputShape(): number[] {
    return this.model.outputShape;
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

  private log(...args: any[]) {
    if (this.verbose) {
      console.log('Federated Client:', ...args);
    }
  }

  private async time(msg: string, action: () => Promise<void>) {
    const t1 = new Date().getTime();
    await action();
    const t2 = new Date().getTime();
    this.log(`${msg} took ${t2 - t1}ms`)
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

// TODO: remove once tfjs >= 0.12.5 is released
function concatWithEmptyTensors(a: tf.Tensor, b: tf.Tensor) {
  if (a.shape[0] === 0) {
    return b.clone();
  } else if (b.shape[0] === 0) {
    return a.clone();
  } else {
    return a.concat(b);
  }
}

function sliceWithEmptyTensors(a: tf.Tensor, begin: number, size?: number) {
  if (begin >= a.shape[0]) {
    return tf.tensor([], [0].concat(a.shape.slice(1)));
  } else {
    return a.slice(begin, size);
  }
}

function addRows(existing: tf.Tensor, newEls: tf.Tensor, unitShape: number[]) {
  if (tf.util.arraysEqual(newEls.shape, unitShape)) {
    return tf.tidy(() => concatWithEmptyTensors(
      existing, tf.expandDims(newEls)));
  } else { // batch dimension
    tf.util.assertShapesMatch(newEls.shape.slice(1), unitShape);
    return tf.tidy(() => concatWithEmptyTensors(existing, newEls));
  }
}
