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

import * as express from 'express';
import * as basicAuth from 'express-basic-auth';
import * as fileUpload from 'express-fileupload';
import {ServerAPI} from 'federated-learning-server';
import {log, verbose} from 'federated-learning-server';
import * as fs from 'fs';
import * as http from 'http';
import * as https from 'https';
import * as path from 'path';
import * as io from 'socket.io';
import * as uuid from 'uuid/v4';

import {labelNames, loadAudioTransferLearningModel} from './model';

const rootDir = path.resolve(__dirname + '/data');
const dataDir = path.join(rootDir, 'data');
const fileDir = path.join(rootDir, 'files');
const modelDir = path.join(rootDir, 'models');
const mkdir = (dir) => !fs.existsSync(dir) && fs.mkdirSync(dir);
[rootDir, modelDir, fileDir, dataDir].forEach(mkdir);
for (let i = 0; i < labelNames.length; i++) {
  mkdir(path.join(fileDir, labelNames[i]));
}

const app = express();

// Use either HTTP or HTTPS, depending on environment variables
let httpServer;
let port;
if (process.env.SSL_KEY && process.env.SSL_CERT) {
  const httpsOptions = {
    key: fs.readFileSync(process.env.SSL_KEY),
    cert: fs.readFileSync(process.env.SSL_CERT)
  };
  httpServer = https.createServer(httpsOptions, app);
  port = process.env.PORT || 443;
} else {
  httpServer = http.createServer(app);
  port = process.env.PORT || 3000;
}

// Set up websockets
const sockServer = io(httpServer);

// Optionally use basic auth
if (process.env.BASIC_AUTH_USER && process.env.BASIC_AUTH_PASS) {
  const users = {};
  users[process.env.BASIC_AUTH_USER] = process.env.BASIC_AUTH_PASS;
  app.use(basicAuth({users, challenge: true}));
}

app.use(fileUpload());

// tslint:disable-next-line:no-any
app.use((req: any, res: any, next: any) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  next();
});

const dataResults = [];
const existingData = fs.readdirSync(dataDir);
existingData.forEach(fn => {
  const json = fs.readFileSync(`${dataDir}/${fn}`).toString();
  dataResults.push(JSON.parse(json));
});

// tslint:disable-next-line:no-any
app.post('/data', (req: any, res: any) => {
  if (!req.files) {
    return res.status(400).send('Must upload a file');
  }
  const file = req.files.file;
  const fileParts = file.name.split('.');
  const labelName = fileParts[0];
  const extension = fileParts[1];
  const labelDir = path.join(fileDir, labelName);
  const fileId = uuid();
  const filename = path.join(labelDir, `${fileId}.${extension}`);
  file.mv(filename);           // save raw file
  dataResults.push(req.body);  // save metadata
  fs.writeFile(`${dataDir}/${fileId}.json`, JSON.stringify(req.body), log);
  res.send('File uploaded!');
});

// tslint:disable-next-line:no-any
app.get('/data', (req: any, res: any) => {
  res.send(dataResults);
});

app.use(express.static(path.resolve(__dirname + '/dist/client')));

verbose(true);

let url =
    'https://storage.googleapis.com/tfjs-speech-command-model-14w/model.json';
let modelVersion = new Date().getTime().toString();
const existingModels = fs.readdirSync(modelDir);
existingModels.sort();
if (existingModels.length) {
  modelVersion = existingModels[existingModels.length - 1];
  url = `file://${modelDir}/${modelVersion}/model.json`;
}

loadAudioTransferLearningModel(url).then(model => {
  const api = new ServerAPI(model, modelVersion, modelDir, sockServer);
  log(`ServerAPI started up at v${api.modelVersion}`);

  httpServer.listen(port, () => {
    console.log(`listening on ${port}`);
  });
});
