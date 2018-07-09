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

/** Server code */

import './fetch_polyfill';

import {Model} from '@tensorflow/tfjs';
import {Server} from 'socket.io';

import {ServerAPI} from './api';
import {federated, FederatedModel} from './common';
import {ModelDB} from './model_db';

export async function setup(
    io: Server, model: FederatedModel|Model, dataDir: string,
    minUpdates?: number) {
  const modelDB = new ModelDB(dataDir, minUpdates);
  await modelDB.setup(federated(model));

  const api = new ServerAPI(modelDB, io);
  await api.setup();

  return api;
}
