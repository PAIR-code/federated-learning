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

import '@tensorflow/tfjs-node';

import * as tf from '@tensorflow/tfjs';
import * as express from 'express';
import * as fileUpload from 'express-fileupload';
import * as federatedServer from 'federated-learning-server';
import {FederatedModel} from 'federated-learning-server';
import * as fs from 'fs';
import * as http from 'http';
import * as path from 'path';
import * as io from 'socket.io';

/**
 * Implementation of `FederatedModel` designed to wrap a loss function and
 * variables it operates over.
 */
export class FederatedDynamicModel implements FederatedModel {
  vars: tf.Variable[];
  loss: (xs: tf.Tensor, ys: tf.Tensor) => tf.Scalar;
  optimizer: tf.Optimizer;
  /**
   * Construct a new `FederatedDynamicModel` wrapping loss function and
   * variables it operates over.
   *
   * @param vars Variables to upload/download from the server.
   * @param loss Loss function to pass to the optimizer.
   * @param learningRate Options ( {learningRate} ) for the optimizer
   */
  constructor(
      vars: tf.Variable[], loss: (xs: tf.Tensor, ys: tf.Tensor) => tf.Scalar,
      optimzer: tf.Optimizer) {
    this.vars = vars;
    this.optimizer = optimzer;
    this.loss = loss;
  }

  async fit(x: tf.Tensor, y: tf.Tensor): Promise<void> {
    this.optimizer.minimize(() => this.loss(x, y));
  }

  getVars(): tf.Variable[] {
    return this.vars.slice();
  }

  setVars(vals: tf.Tensor[]): void {
    for (let i = 0; i < vals.length; i++) {
      this.vars[i].assign(vals[i]);
    }
  }
}

// tslint:disable:max-line-length
const MODEL_URL =
    'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/web_model.pb';
const WEIGHT_MANIFEST =
    'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/weights_manifest.json'

// Load the model & set it up for training
async function setupModel() {
  const model = await tf.loadFrozenModel(MODEL_URL, WEIGHT_MANIFEST);
  const vars = (model as any).executor.weightMap as {[wn: string]: tf.Tensor[]};

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
const httpServer = http.createServer(app);
const sockServer = io(httpServer);
const port = process.env.PORT || 3000;

app.use(fileUpload());

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  next();
});

app.post('/data', (req, res) => {
  res.status(400).send('file upload TODO');
  /*
  if (!req.files) {
    res.status(400).send('Must upload a file');
  } else {
    res.send('File uploaded!');
  }

  const file = req.files.file;
  const fileParts = file.name.split('.');
  const labelName = fileParts[0];
  const extension = fileParts[1];
  const labelDir = path.join(fileDir, labelName);
  const filename = path.join(labelDir, `${uuid()}.${extension}`);
  file.mv(filename);
  */
});

setupModel().then(({varsAndLoss}) => {
  federatedServer.setup(sockServer, varsAndLoss, dataDir, 1).then(() => {
    mkdir(fileDir);

    httpServer.listen(port, () => {
      console.log(`listening on ${port}`);
    });
  });
});
