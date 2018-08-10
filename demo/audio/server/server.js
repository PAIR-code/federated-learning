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

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const express = require('express');
const basicAuth = require('express-basic-auth');
const fileUpload = require('express-fileupload');
const federated = require('federated-learning-server');
const fs = require('fs');
const http = require('http');
const https = require('https');
const path = require('path');
const uuid = require('uuid/v4');
const util = require('util');
const fetch = require('node-fetch');
const npy = require('tfjs-npy');
const writeFile = util.promisify(fs.writeFile);

// Setup express app using either HTTP or HTTPS, depending on environment vars
const app = express();
let webServer;
let port;
if (process.env.SSL_KEY && process.env.SSL_CERT) {
  const httpsOptions = {
    key: fs.readFileSync(process.env.SSL_KEY),
    cert: fs.readFileSync(process.env.SSL_CERT)
  };
  webServer = https.createServer(httpsOptions, app);
  port = process.env.PORT || 443;
} else {
  webServer = http.createServer(app);
  port = process.env.PORT || 3000;
}

// Optionally use basic auth
if (process.env.BASIC_AUTH_USER && process.env.BASIC_AUTH_PASS) {
  const users = {};
  users[process.env.BASIC_AUTH_USER] = process.env.BASIC_AUTH_PASS;
  app.use(basicAuth({users, challenge: true}));
}

// Setup file uploading (to save .wav files)
app.use(fileUpload());

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  next();
});

// Create directories
const labelNames = ['accio', 'expelliarmus', 'lumos', 'nox'];
const rootDir = path.resolve(__dirname + '/data');
const fileDir = path.join(rootDir, 'files');
const modelDir = path.join(rootDir, 'models');
const mkdir = (dir) => !fs.existsSync(dir) && fs.mkdirSync(dir);
[rootDir, modelDir, fileDir].forEach(mkdir);
for (let i = 0; i < labelNames.length; i++) {
  mkdir(path.join(fileDir, labelNames[i]));
}

// Load existing performance metrics
const metrics = {};
const metricsPath = (version) => path.join(modelDir, version, 'metrics.json');
fs.readdirSync(modelDir).forEach(v => {
  if (fs.existsSync(metricsPath(v))) {
    metrics[v] = JSON.parse(fs.readFileSync(metricsPath(v)).toString());
  }
});

// Setup endpoints to track client and validation accuracy for visualization
app.post('/data', (req, res) => {
  // Record metrics
  const version = req.body.modelVersion;
  const clientId = req.body.clientId;
  if (!metrics[version]['clients'][clientId]) {
    metrics[version]['clients'][clientId] = [];
  }
  metrics[version]['clients'][clientId].push(JSON.parse(req.body.metrics));

  // Save files + metadata for later analysis (dont't do this in real life)
  const reqId = `${clientId}_${uuid()}`;
  const wavFile = req.files.wav;
  const npyFile = req.files.npy;
  const labelName = wavFile.name.split('.')[0];
  wavFile.mv(`${fileDir}/${labelName}/${reqId}.wav`, () => {});
  npyFile.mv(`${fileDir}/${labelName}/${reqId}.npy`, () => {});
  writeFile(`${fileDir}/${labelName}/${reqId}.json`, JSON.stringify(req.body));

  res.sendStatus(200);
});

// Expose in-memory metrics
app.get('/metrics', async (req, res) => {
  res.send(metrics);
});

// Expose the client as a set of static files
app.use(express.static(path.resolve(__dirname + '/dist/client')));

const initialModelUrl =
  'https://storage.googleapis.com/tfjs-speech-command-model-14w/model.json';
const validInputUrl =
  'https://storage.googleapis.com/tfjs-federated-hogwarts/val-inputs.npy';
const validLabelUrl =
  'https://storage.googleapis.com/tfjs-federated-hogwarts/val-labels.npy';

async function loadNpyUrl(url) {
  const res = await fetch(url);
  const arr = await res.arrayBuffer();
  return npy.parse(arr);
}

async function loadInitialModel() {
  const model = await tf.loadModel(initialModelUrl);
  for (let i = 0; i < model.layers.length; ++i) {
    model.layers[i].trainable = false;  // freeze everything
  }
  const transferLayer = model.layers[10].output;
  const newDenseLayer = tf.layers.dense({units: 4, activation: 'softmax'});
  const newOutputs = newDenseLayer.apply(transferLayer);
  return tf.model({inputs: model.inputs, outputs: newOutputs});
}

const fedServer = new federated.Server(webServer, loadInitialModel, {
  modelDir,
  updatesPerVersion: 5,
  clientHyperparams: {
    examplesPerUpdate: 4,
    epochs: 10,
    learningRate: 0.001,
    weightNoiseStddev: 0.0001
  },
  modelCompileConfig: {
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  },
  verbose: true
});

async function setup() {
  const validInputs = await loadNpyUrl(validInputUrl);
  const validLabelsFlat = await loadNpyUrl(validLabelUrl);
  const validLabels = tf.oneHot(validLabelsFlat, 4);
  tf.dispose(validLabelsFlat);

  fedServer.onNewVersion((model, oldVersion, newVersion) => {
    // Save old metrics to disk, now that we're done with them
    if (metrics[oldVersion]) {
      writeFile(metricsPath(oldVersion), JSON.stringify(metrics[oldVersion]));
    }
    // Create space for the new model's metrics
    if (!metrics[newVersion]) {
      metrics[newVersion] = {validation: null, clients: {}}
    }
    // Compute validation accuracy
    const newValMetrics = model.evaluate(validInputs, validLabels);
    metrics[newVersion].validation = newValMetrics;
    console.log(`Version ${newVersion} validation metrics: ${newValMetrics}`);
  });

  await fedServer.setup();

  webServer.listen(port, () => {
    console.log(`listening on ${port}`);
  });
}

setup();
