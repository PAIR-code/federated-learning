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

// tslint:disable-next-line:no-any
type MockitMsg = any;
// tslint:disable-next-line:no-any
type MockitCallback = (msg: MockitMsg, ack?: any) => void|Promise<void>;
type MockitCallbackDict = {
  [key: string]: Function
};
type ClientDict = {
  [key: string]: WrappedClient
};
const CONNECTION_TIMEOUT = 10;

class WrappedClient {
  client: MockitIOClient;
  listeners: MockitCallbackDict;
  constructor(client: MockitIOClient) {
    this.client = client;
    this.listeners = {};
  }

  async trigger(event: string, msg: MockitMsg) {
    if (this.listeners[event]) {
      this.listeners[event](msg);
    }
  }

  on(event: string, callback: MockitCallback) {
    this.listeners[event] = callback;
  }

  emit(event: string, msg: MockitMsg) {
    this.client.trigger(event, msg);
  }
}

export class MockitIOServer {
  clients: ClientDict;

  constructor() {
    this.clients = {};
  }

  onConnect: (socket: WrappedClient) => void;

  on(event: string, listener: (socket: WrappedClient) => void) {
    if (event !== 'connection') {
      throw new Error(`this mock doesn't support non-connection evts`);
    }
    this.onConnect = listener;
  }

  connect(client: MockitIOClient) {
    const wrapped = new WrappedClient(client);
    this.clients[client.clientId] = wrapped;
    this.onConnect(wrapped);
  }

  findClient(client: MockitIOClient): WrappedClient {
    return this.clients[client.clientId];
  }

  get sockets() {
    return this;
  }

  emit(event: string, msg: MockitMsg) {
    for (const k in this.clients) {
      this.clients[k].emit(event, msg);
    }
  }
}

export class MockitIOClient {
  server: MockitIOServer;
  listeners: MockitCallbackDict;
  clientId: string;

  constructor(server: MockitIOServer, id: string) {
    this.server = server;
    this.listeners = {};
    this.clientId = id;
    setTimeout(() => this.server.connect(this), CONNECTION_TIMEOUT);
  }

  trigger(event: string, msg: MockitMsg) {
    if (this.listeners[event]) {
      this.listeners[event](msg);
    }
  }

  // tslint:disable-next-line:no-any
  emit(event: string, ...args: any[]) {
    const message = args[0] as string;
    const callback = args[1] as MockitCallback;
    this.server.findClient(this).trigger(event, message).then(callback);
  }

  on(event: string, listener: Function) {
    this.listeners[event] = listener;
  }

  removeListener(event: string, listener: Function) {
    console.log(`removing ${listener}`);
    this.listeners[event] = null;
  }

  disconnect() {
    this.server = null;
  }
}
