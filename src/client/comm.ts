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
import {ModelFitConfig, Variable} from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {LayerVariable} from '@tensorflow/tfjs-layers/dist/variables';
import * as socketio from 'socket.io-client';

import {ConnectionMsg, Events, VarsMsg} from '../common';
// tslint:disable-next-line:max-line-length
import {deserializeVar, SerializedVariable, serializeVar} from '../serialization';

const CONNECTION_TIMEOUT = 10 * 1000;
const UPLOAD_TIMEOUT = 1 * 1000;

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
export class VariableSynchroniser {
  public modelId: string;
  private socket: SocketIOClient.Socket;
  private connMsg: ConnectionMsg;
  private vars = new Map<string, Variable|LayerVariable>();
  private acceptUpdate: (msg: VarsMsg) => boolean;
  /**
   * Construct a synchroniser from a list of tf.Variables of tf.LayerVariables.
   * @param {Array<Variable|LayerVariable>} vars - Variables to track and sync
   */
  constructor(
      vars: Array<Variable|LayerVariable>,
      updateCallback?: (msg: VarsMsg) => boolean) {
    for (const variable of vars) {
      this.vars.set(variable.name, variable);
    }
    if (updateCallback) {
      this.acceptUpdate = updateCallback;
    } else {
      this.acceptUpdate = () => true;
    }
  }

  /**
   * Construct a VariableSynchroniser from an array of layers.
   * This will synchronise the weights of the layers.
   * @param layers: An array of layers to extract variables from
   */
  public static fromLayers(layers: Layer[]) {
    const layerWeights = layers.map(l => l.trainableWeights);
    return new VariableSynchroniser(tf.util.flatten(layerWeights, []));
  }

  private async connect(url: string): Promise<ConnectionMsg> {
    this.socket = socketio(url);
    return fromEvent<ConnectionMsg>(
        this.socket, Events.Connect, CONNECTION_TIMEOUT);
  }

  /**
   * Connect to a server, synchronise the variables to their
   * initial values and return the hyperparameters for this client
   * @param url: The URL of the server
   * @return A promise that resolves when the connection has been established
   * and variables set to their inital values.
   */
  public async initialise(url: string): Promise<ModelFitConfig> {
    this.connMsg = await this.connect(url);
    this.setVarsFromMessage(this.connMsg.initVars);
    this.modelId = this.connMsg.modelId;

    this.socket.on(Events.Download, (msg: VarsMsg) => {
      if (this.acceptUpdate(msg)) {
        this.setVarsFromMessage(msg.vars);
        this.modelId = msg.modelId;
      }
    });

    return this.connMsg.fitConfig;
  }

  /**
   * Upload the current values of the tracked variables to the server
   * @return A promise that resolves when the server has recieved the variables
   */
  public async uploadVars(): Promise<{}> {
    const msg: VarsMsg = await this.serializeCurrentVars();
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

  protected async serializeCurrentVars(): Promise<VarsMsg> {
    const varsP: Array<Promise<SerializedVariable>> = [];

    this.vars.forEach((value, key) => {
      if (value instanceof LayerVariable) {
        varsP.push(serializeVar(tf.variable(value.read())));
      } else {
        varsP.push(serializeVar(value));
      }
    });
    const vars = await Promise.all(varsP);
    return {clientId: this.connMsg.clientId, modelId: this.modelId, vars};
  }

  protected setVarsFromMessage(newVars: SerializedVariable[]) {
    for (const param of newVars) {
      if (!this.vars.has(param.name)) {
        throw new Error(`Recieved message with unexpected param ${
            param.name}, should be one of ${this.vars.keys()}`);
      }
      const varOrLVar = this.vars.get(param.name);
      if (varOrLVar instanceof LayerVariable) {
        varOrLVar.write(deserializeVar(param));
      } else {
        varOrLVar.assign(deserializeVar(param));
      }
    }
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
