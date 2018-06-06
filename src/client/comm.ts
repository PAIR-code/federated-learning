import * as tf from '@tensorflow/tfjs';
import {ModelFitConfig, Variable} from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {LayerVariable} from '@tensorflow/tfjs-layers/dist/variables';
import * as socketio from 'socket.io-client';

import {ConnectionMsg, Events, VarsMsg} from '../common';

const CONNECTION_TIMEOUT = 10 * 1000;
const UPLOAD_TIMEOUT = 1 * 1000;

async function fromEvent<T>(
    emitter: SocketIOClient.Socket, eventName: string, timeout: number) {
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

function flatten<T>(arr: T[][]) {
  return arr.reduce((x, y) => x.concat(y), []);
}

export class VariableSynchroniser {
  public version: string;
  private socket: SocketIOClient.Socket;
  private connMsg: ConnectionMsg;
  private vars: Map<string, Variable|LayerVariable>;

  constructor(syncVars: Array<Variable|LayerVariable>) {
    for (const param of syncVars) {
      this.vars.set(param.name, param);
    }
  }

  public static fromLayers(layers: Layer[]) {
    const layerWeights = layers.map(l => l.trainableWeights);
    return new VariableSynchroniser(flatten(layerWeights));
  }

  private async connect(url: string): Promise<ConnectionMsg> {
    this.socket = socketio(url);

    return fromEvent<ConnectionMsg>(
        this.socket, Events.Connect, CONNECTION_TIMEOUT);
  }

  public async initialise(url: string): Promise<ModelFitConfig> {
    this.connMsg = await this.connect(url);
    await this.setVarsFromMessage(this.connMsg.initVars);
    this.version = this.connMsg.version;

    this.socket.on(Events.Download, (msg: VarsMsg) => {
      this.setVarsFromMessage(msg.vars);
      this.version = msg.version;
    });

    return this.connMsg.fitConfig;
  }

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
    const vars: Variable[] = [];

    this.vars.forEach((value, key) => {
      if (value instanceof LayerVariable) {
        vars.push(tf.variable(value.read()));
      } else {
        vars.push(value);
      }
    });

    return {clientId: this.connMsg.clientId, version: this.version, vars};
  }

  protected async setVarsFromMessage(newVars: Variable[]) {
    for (const param of newVars) {
      if (!this.vars.has(param.name)) {
        throw new Error(`Recieved message with unexpected param ${
            param.name}, should be one of ${this.vars.keys()}`);
      }
      const varOrLVar = this.vars.get(param.name);
      if (varOrLVar instanceof LayerVariable) {
        varOrLVar.write(param);
      } else {
        varOrLVar.assign(param);
      }
    }
  }
}
