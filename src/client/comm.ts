import * as tf from '@tensorflow/tfjs';
import {ModelFitConfig, Variable} from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import {LayerVariable} from '@tensorflow/tfjs-layers/dist/variables';
import * as socketio from 'socket.io-client';

import {ConnectionMsg, Events, ParamsMsg} from '../common';

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

type ParamMap = {
  [key: string]: Variable
};

function flatten<T>(arr: T[][]) {
  return arr.reduce((x, y) => x.concat(y), []);
}

class ParamsSynchroniser {
  public version: string;
  private socket: SocketIOClient.Socket;
  private connMsg: ConnectionMsg;
  private params: Map<string, Variable|LayerVariable>;

  constructor(syncParams: Array<Variable|LayerVariable>) {
    for (const param of syncParams) {
      this.params.set(param.name, param);
    }
  }

  public static fromLayers(layers: Layer[]) {
    const layerWeights = layers.map(l => l.trainableWeights);
    return new ParamsSynchroniser(flatten(layerWeights));
  }

  private async connect(url: string): Promise<ConnectionMsg> {
    this.socket = socketio(url);

    return fromEvent<ConnectionMsg>(
        this.socket, Events.Connect, CONNECTION_TIMEOUT);
  }

  public async initialise(url: string): Promise<ModelFitConfig> {
    this.connMsg = await this.connect(url);
    await this.setParamsFromMessage(this.connMsg.initParams);
    this.version = this.connMsg.version;

    this.socket.on(Events.Download, (msg: ParamsMsg) => {
      this.setParamsFromMessage(msg.params);
      this.version = msg.version;
    });

    return this.connMsg.fitConfig;
  }

  public async uploadParams(): Promise<{}> {
    const msg: ParamsMsg = await this.serializeCurrentParams();
    const prom = new Promise((resolve, reject) => {
      const rejectTimer =
          setTimeout(() => reject(`uploadParams timed out`), UPLOAD_TIMEOUT);

      this.socket.emit(Events.Upload, msg, () => {
        clearTimeout(rejectTimer);
        resolve();
      });
    });
    return prom;
  }

  protected async serializeCurrentParams() {
    const params: Variable[] = [];

    this.params.forEach((value, key) => {
      if (value instanceof LayerVariable) {
        params.push(tf.variable(value.read()));
      } else {
        params.push(value);
      }
    });

    return {clientId: this.connMsg.clientId, version: this.version, params};
  }

  protected async setParamsFromMessage(newParams: Variable[]) {
    for (const param of newParams) {
      if (!this.params.has(param.name)) {
        throw new Error(`Recieved message with unexpected param ${
            param.name}, should be one of ${this.params.keys()}`);
      }
      const varOrLVar = this.params.get(param.name);
      if (varOrLVar instanceof LayerVariable) {
        varOrLVar.write(param);
      } else {
        varOrLVar.assign(param);
      }
    }
  }
}

async function exampleUsageToSilenceTSLint() {
  const m = await tf.loadModel('hello.model');
  const l = ParamsSynchroniser.fromLayers([m.getLayer('hello.layer')]);
  const fitConfig = await l.initialise('hello.com');
  m.fit(tf.ones([1]), tf.ones([1]), fitConfig);
  await l.uploadParams();
}

exampleUsageToSilenceTSLint();
