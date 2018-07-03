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

import * as expressProxy from 'express';
import * as fileUploadProxy from 'express-fileupload';
import * as fs from 'fs';
import * as http from 'http';
import * as path from 'path';
import * as socketProxy from 'socket.io';
import {promisify} from 'util';
import * as uuidProxy from 'uuid/v4';

// tslint:disable-next-line:no-angle-bracket-type-assertion no-any
const uuid = (<any>uuidProxy).default || uuidProxy;
// tslint:disable-next-line:no-angle-bracket-type-assertion no-any
const express = (<any>expressProxy).default || expressProxy;
// tslint:disable-next-line:no-angle-bracket-type-assertion no-any
const fileUpload = (<any>fileUploadProxy).default || fileUploadProxy;
// tslint:disable-next-line:no-angle-bracket-type-assertion no-any
const socketIO = (<any>socketProxy).default || socketProxy;

import {FederatedModel} from './common';
import {ServerAPI} from './api';
import {ModelDB} from './model_db';

const mkdir = promisify(fs.mkdir);
const exists = promisify(fs.exists);

export async function setup(model: FederatedModel, dataDir: string) {
  const app = express();
  const server = http.createServer(app);
  const io = socketIO(server);
  const modelDB = new ModelDB(dataDir);
  const api = new ServerAPI(modelDB, io);

  app.use(fileUpload());

  app.use((req: any, res: any, next: any) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    next();
  });

  // tslint:disable-next-line:no-any
  app.post('/data', async (req: any, res: any) => {
    if (!req.files) {
      return res.status(400).send('Must upload a file');
    }
    const dataPath = path.join(dataDir, 'files');
    const dirExists = await exists(dataPath);
    if (!dirExists) {
      await mkdir(dataPath);
    }
    const file = req.files.file;
    const filename = path.join(dataPath, uuid() + '_' + file.name);
    await file.mv(filename);
    res.send('File uploaded successfully');
  });

  return modelDB.setup(model).then(() => {
    api.setup().then(() => {
      server.listen(3000, () => {
        console.log('listening on 3000');
      });
    });
  });
}
