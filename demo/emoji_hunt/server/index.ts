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

// tslint:disable:max-line-length
const MODEL_URL =
    'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/web_model.pb';
const WEIGHT_MANIFEST =
    'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/weights_manifest.json';

import '@tensorflow/tfjs-node';

import * as tf from '@tensorflow/tfjs';
import * as express from 'express';
import * as fileUpload from 'express-fileupload';
import * as federatedServer from 'federated-learning-server';
import {FederatedDynamicModel} from 'federated-learning-server';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';
import * as fs from 'fs';
// import * as http from 'http';
import * as path from 'path';
import * as io from 'socket.io';

// Load the model & set it up for training
async function setupModel() {
  const model = await loadFrozenModel(MODEL_URL, WEIGHT_MANIFEST);
  const vars = model.weights;

  // TODO: there must be a better way
  const nonTrainables = /(batchnorm)|(reshape)/g;

  // Make weights trainable & extract them
  const trainable: tf.Variable[] = [];

  for (const weightName in vars) {
    if (!weightName.match(nonTrainables)) {
      vars[weightName] = vars[weightName].map((t: tf.Tensor) => {
        if (t.dtype === 'float32') {
          const ret = tf.variable(t);
          trainable.push(ret);
          return ret;
        } else {
          return t;
        }
      });
    }
  }

  // TODO: better to not run softmax and use softmaxCrossEntropy?
  const loss = (input: tf.Tensor, label: tf.Tensor) => {
    const preds = model.predict(input) as tf.Tensor;
    return tf.losses.logLoss(label, preds) as tf.Scalar;
  };

  const optimizer = tf.train.sgd(0.1);

  const varsAndLoss = new FederatedDynamicModel(trainable, loss, optimizer);

  return {model, varsAndLoss};
}

const dataDir = path.resolve(__dirname + '/data');
const fileDir = path.join(dataDir, 'files');
const mkdir = (dir: string) => !fs.existsSync(dir) && fs.mkdirSync(dir);

const app = express();
// const httpServer = http.createServer(app);
const sockServer = io({secure: true});
const port = 3000;

app.use(fileUpload());

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  next();
});

app.post('/data', (req, res) => {
  res.status(400).send('file upload TODO');
});

setupModel().then(({varsAndLoss}) => {
  federatedServer.setup(sockServer, varsAndLoss, dataDir, 1).then(() => {
    mkdir(fileDir);
    sockServer.listen(port);
  });
});
