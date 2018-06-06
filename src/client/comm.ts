import * as tf from '@tensorflow/tfjs';
import {ModelFitConfig} from '@tensorflow/tfjs';
import {Layer} from '@tensorflow/tfjs-layers/dist/engine/topology';
import * as socketio from 'socket.io-client';
import {Events} from '../common';

const CONNECTION_TIMEOUT = 10 * 1000;

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

class LayersSynchroniser {
  private socket: SocketIOClient.Socket;
  constructor(private layers: Layer[]) {}

  public async connect(url: string): Promise<ConnectionInfo> {
    this.socket = socketio(url);

    return fromEvent<ConnectionInfo>(
        this.socket, 'connect', CONNECTION_TIMEOUT);
  }

  public async uploadParams(): Promise<void> {
    this.socket.emit(Events.Upload, {'version'})
    return Promise.resolve();
  }
}

async function exampleUsageToSilenceTSLint() {
  const m = await tf.loadModel('hello.model');
  const l = new LayersSynchroniser([m.getLayer('hello.layer')]);
  l.connect('hello.com');
  // train m
  await l.uploadParams();
}

main();
