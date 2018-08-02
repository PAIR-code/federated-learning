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
import {HyperparamsMsg, ServerAPI} from 'federated-learning-server';
import {log} from 'federated-learning-server';
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
const metaDir = path.join(rootDir, 'metadata');
const fileDir = path.join(rootDir, 'files');
const modelDir = path.join(rootDir, 'models');
const mkdir = (dir) => !fs.existsSync(dir) && fs.mkdirSync(dir);
[rootDir, modelDir, fileDir, metaDir].forEach(mkdir);
for (let i = 0; i < labelNames.length; i++) {
  mkdir(path.join(fileDir, labelNames[i]));
}

// Load metadata about our data / client-side accuracy
const metadata = fs.readdirSync(metaDir).map(
    file => JSON.parse(fs.readFileSync(`${metaDir}/${file}`).toString()));

// Load validation sets + data about validation accuracy
function parseNpyFile(name): tf.Tensor {
  const buff = fs.readFileSync(path.resolve(__dirname + '/' + name));
  const arrayBuff =
      buff.buffer.slice(buff.byteOffset, buff.byteOffset + buff.byteLength);
  return npy.parse(arrayBuff);
}

const validInputs = parseNpyFile('hp-validation-inputs.npy');
const validLabels = tf.tidy(
    () =>
        tf.oneHot(parseNpyFile('hp-validation-labels.npy') as tf.Tensor1D, 4));

// Setup endpoints to track client and validation accuracy for visualization

// tslint:disable-next-line:no-any
app.post('/data', (req: any, res: any) => {
  const reqId = uuid();

  metadata.push(req.body);  // save metadata
  writeFile(`${metaDir}/${reqId}.json`, JSON.stringify(req.body));

  if (req.files) {
    req.files.forEach(file => {
      const fileParts = file.name.split('.');
      const labelName = fileParts[0];
      const extension = fileParts[fileParts.length - 1];
      file.mv(`${fileDir}/${labelName}/${reqId}.${extension}`);
    });
  }

  res.send(200);
});

// tslint:disable-next-line:no-any
app.get('/metadata', async (req: any, res: any) => {
  res.send(metadata);
});

// tslint:disable-next-line:no-any
app.get('/validation', (req: any, res: any) => {
  res.send(validation);
})

// Expose the client as a set of static files
app.use(express.static(path.resolve(__dirname + '/dist/client')));

// Either load our model from the internet or our data directory

const hyperparams: HyperparamsMsg = {
  examplesPerUpdate: 4,
  epochs: 10,
  updatesPerVersion: 5
};

const initialModelUrl =
    'https://storage.googleapis.com/tfjs-speech-command-model-14w/model.json';

async function loadInitialModel(): Promise<tf.Model> {
  const model = await tf.loadModel(initialModelUrl);
  for (let i = 0; i < model.layers.length; ++i) {
    model.layers[i].trainable = false;  // freeze everything
  }
  const transferLayer = model.layers[10].output;
  const newDenseLayer = tf.layers.dense({units: 4, activation: 'softmax'});
  const newOutputs = newDenseLayer.apply(transferLayer) as tf.SymbolicTensor;
  return tf.model({inputs: model.inputs, outputs: newOutputs});
}


const federatedModel =
    new ServerTfModel(modelDir, loadInitialModel, compileModel);

loadAudioTransferLearningModel(url).then(model => {
  // Setup our federated learning API
  const api = new ServerAPI(sockServer, federatedModel, {
    clientHyperparams: hyperparams,
    validationData: [validInputs, validLabels]
  });
  log(`ServerAPI started up at v${api.modelVersion}`);

  // Add a callback whenever the model is updated to compute validation accuracy
  const updateResults = (fed) => {
    if (!validResults[fed.version]) {
      const results = tf.tidy(() => {
        const r = model.evaluate(validInputs, validLabels);
        return [r[0].dataSync()[0], r[1].dataSync()[0]];
      });
      validResults[fed.version] = {
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
