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

import * as tf from '@tensorflow/tfjs';
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
import * as npy from './npy';

// Setup express app using either HTTP or HTTPS, depending on environment
// variables
const app = express();
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

// Load tfjs-node (below other code, so clang-format doesn't move it)
import '@tensorflow/tfjs-node';

// Set up websockets
const sockServer = io(httpServer);

// Optionally use basic auth
if (process.env.BASIC_AUTH_USER && process.env.BASIC_AUTH_PASS) {
  const users = {};
  users[process.env.BASIC_AUTH_USER] = process.env.BASIC_AUTH_PASS;
  app.use(basicAuth({users, challenge: true}));
}

// Setup file uploading (to save .wav files)
app.use(fileUpload());
// tslint:disable-next-line:no-any
app.use((req: any, res: any, next: any) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  next();
});

// Create directories
const rootDir = path.resolve(__dirname + '/data');
const dataDir = path.join(rootDir, 'data');
const fileDir = path.join(rootDir, 'files');
const modelDir = path.join(rootDir, 'models');
const mkdir = (dir) => !fs.existsSync(dir) && fs.mkdirSync(dir);
[rootDir, modelDir, fileDir, dataDir].forEach(mkdir);
for (let i = 0; i < labelNames.length; i++) {
  mkdir(path.join(fileDir, labelNames[i]));
}

// Load metadata about our data / client-side accuracy
const dataResults = [];
const existingData = fs.readdirSync(dataDir);
existingData.forEach(fn => {
  const json = fs.readFileSync(`${dataDir}/${fn}`).toString();
  dataResults.push(JSON.parse(json));
});

// Load validation sets + data about validation accuracy
function parseNpyFile(name): tf.Tensor {
  const buff = fs.readFileSync(path.resolve(__dirname + '/' + name));
  const arrayBuff =
      buff.buffer.slice(buff.byteOffset, buff.byteOffset + buff.byteLength);
  return npy.parse(arrayBuff);
}
const validInputs = parseNpyFile('validation-inputs.npy');
const validLabels = tf.tidy(
    () => tf.oneHot(parseNpyFile('validation-labels.npy') as tf.Tensor1D, 14));
const validResultPath = `${rootDir}/validation.json`;
let validResults = {};
if (fs.existsSync(validResultPath)) {
  validResults = JSON.parse(fs.readFileSync(validResultPath).toString());
}

// Setup endpoints to track client and validation accuracy for visualization

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
  fs.writeFile(`${dataDir}/${fileId}.json`, JSON.stringify(req.body), () => {});
  res.send('File uploaded!');
});

// tslint:disable-next-line:no-any
app.get('/data', (req: any, res: any) => {
  res.send(dataResults);
});

// tslint:disable-next-line:no-any
app.get('/validation', (req: any, res: any) => {
  res.send(validResults);
})

// Expose the client as a set of static files
app.use(express.static(path.resolve(__dirname + '/dist/client')));

// Either load our model from the internet or our data directory
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
  // Setup our federated learning API
  const api = new ServerAPI(model, modelVersion, modelDir, sockServer);
  log(`ServerAPI started up at v${api.modelVersion}`);

  // Add a callback whenever the model is updated to compute validation accuracy
  const updateResults = (modelVersion) => {
    if (!validResults[modelVersion]) {
      const results = tf.tidy(() => {
        const r = model.evaluate(validInputs, validLabels);
        return [r[0].dataSync()[0], r[1].dataSync()[0]];
      });
      validResults[modelVersion] = {
        'Cross-Entropy': results[0],
        'Accuracy': results[1]
      };
      fs.writeFile(validResultPath, JSON.stringify(validResults), () => {});
    }
  };
  api.onUpdate(updateResults);
  updateResults(modelVersion);

  // Finally start listening for clients
  httpServer.listen(port, () => {
    console.log(`listening on ${port}`);
  });
});
