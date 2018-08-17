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

import {Server as IOServer} from 'socket.io';

// tslint:disable-next-line:max-line-length
import {AbstractServer, FederatedServerConfig, FederatedServerInMemoryModel, FederatedServerModel, isFederatedServerModel} from './abstract_server';
import {AsyncTfModel} from './common';
import {MockitIOClient, MockitIOServer} from './mockit_io';

/**
 * Mocked out version of the federated server library that can be used for
 * in-browser emulation of federated learning (for demo and dev purposes).
 *
 * Example usage:
 * ```js
 * import {Server as FederatedServer} from 'federated-learning-mock-server';
 * import {Client as FederatedClient} from 'federated-learning-client';
 *
 * const model = await tf.loadModel('https://my.model.json');
 *
 * const server = new FederatedServer(model);
 * await server.setup();
 *
 * const client = new FederatedClient(server.newClientSocket, model);
 * await client.setup();
 * ```
 */
export class Server extends AbstractServer {
  newClientSocket: () => SocketIOClient.Socket;

  constructor(
      model: AsyncTfModel|FederatedServerModel, config: FederatedServerConfig) {
    const server = new MockitIOServer();

    let fedModel: FederatedServerModel;
    if (isFederatedServerModel(model)) {
      fedModel = model;
    } else {
      const compileConfig = config.modelCompileConfig || {};
      fedModel = new FederatedServerInMemoryModel(model, compileConfig);
    }

    // tslint:disable-next-line:no-any
    const ioServer = (server as any) as IOServer;
    super(ioServer, fedModel, config);

    this.newClientSocket = () => {
      const socket = new MockitIOClient(server, uuid());
      // tslint:disable-next-line:no-any
      return (socket as any) as SocketIOClient.Socket;
    };
  }
}

function uuid() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}
