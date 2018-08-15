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

import * as http from 'http';
import * as https from 'https';
import * as path from 'path';
import * as io from 'socket.io';

// tslint:disable-next-line:max-line-length
import {AbstractServer, FederatedServerConfig, FederatedServerModel, isFederatedServerModel} from './abstract_server';
import {AsyncTfModel} from './common';
import {FederatedServerTfModel} from './models';

/**
 * Federated Learning Server library.
 *
 * Example usage with a tf.Model:
 * ```js
 * const model = await tf.loadModel('file:///a/model.json');
 * const webServer = http.createServer();
 * const fedServer = new Server(webServer, model);
 * fedServer.setup().then(() => {
 *  webServer.listen(80);
 * });
 * ```
 *
 * The server aggregates model weight updates from clients and publishes new
 * versions of the model periodically to all clients.
 */
export class Server extends AbstractServer {
  constructor(
      server: http.Server|https.Server|io.Server,
      model: AsyncTfModel|FederatedServerModel, config: FederatedServerConfig) {
    // Setup server
    let ioServer = server;
    if (server instanceof http.Server || server instanceof https.Server) {
      ioServer = io(server);
    }

    // Setup model
    let fedModel = model;
    if (!isFederatedServerModel(model)) {
      const defaultDir = path.resolve(`${process.cwd()}/saved-models`);
      const modelDir = config.modelDir || defaultDir;
      const compileConfig = config.modelCompileConfig || {};
      fedModel = new FederatedServerTfModel(modelDir, model, compileConfig);
    }

    if (!config.verbose) {
      config.verbose = (!!process.env.VERBOSE);
    }

    super(ioServer as io.Server, fedModel as FederatedServerModel, config);
  }
}
