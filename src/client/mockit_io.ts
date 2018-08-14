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
type MockitCallback = (msg: MockitMsg) => Promise<void>;
type MockitCallbackDict = {
  [key: string]: MockitCallback
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
  clients: WrappedClient[];
  onConnect: (socket: WrappedClient) => Promise<void>;

  on(event: string, listener: (socket: WrappedClient) => Promise<void>) {
    if (event !== 'connection') {
      throw new Error(`this mock doesn't support non-connection evts`);
    }
    this.onConnect = listener;
  }

  connect(client: MockitIOClient) {
    const wrapped = new WrappedClient(client);
    this.clients.push(wrapped);
    this.onConnect(wrapped);
  }

  findClient(client: MockitIOClient): WrappedClient {
    // TODO
    return this.clients[0];
  }

  get sockets() {
    return this;
  }

  emit(event: string, msg: MockitMsg) {
    this.clients.forEach(c => c.emit(event, msg));
  }
}

export class MockitIOClient {
  server: MockitIOServer;
  listeners: MockitCallbackDict;

  constructor(server: MockitIOServer) {
    this.server = server;
    this.listeners = {};
    setTimeout(() => {
      this.server.connect(this);
    }, CONNECTION_TIMEOUT);
  }

  trigger(event: string, msg: MockitMsg) {
    if (this.listeners[event]) {
      this.listeners[event](msg);
    }
  }

  emit(event: string, message: MockitMsg, callback: MockitCallback) {
    this.server.findClient(this).trigger(event, message).then(callback);
  }

  on(event: string, listener: MockitCallback) {
    this.listeners[event] = listener;
  }

  removeListener(event: string) {
    this.listeners[event] = null;
  }
}
