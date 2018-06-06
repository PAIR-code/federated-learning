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

import * as tf from '@tensorflow/tfjs';
import * as express from 'express';
import {Request, Response} from 'express';
import * as fs from 'fs';
import * as http from 'http';
import * as path from 'path';
import * as socketIO from 'socket.io';
import * as uuid from 'uuid/v4';

const app = express();
const server = http.createServer(app);
const io = socketIO(server);

const indexPath = path.resolve(__dirname + '/../demo/index.html');
const dataDir = path.resolve(__dirname + '/../data/');
const MIN_UPDATES = 20;
let currentModelId = '';

async function possiblyUpdateModel(modelId: string) {
  const modelDir = path.join(dataDir, modelId);
  const modelPath = path.join(dataDir, modelId + '.json');
  const updateFiles = fs.readdirSync(modelDir);
  if (updateFiles.length > MIN_UPDATES) {
    const currentJSON = JSON.parse(fs.readFileSync(modelPath).toString());
    const updatesJSON = updateFiles.map((filename) => {
      return JSON.parse(fs.readFileSync(filename).toString());
    });
    const updatedVars: tf.Tensor[] = currentJSON['variables'].map(
        (variable: {values: number[], shape: number[]}) => {
          return tf.tensor(variable.values, variable.shape);
        });
    let numExamples = 0;
    updatesJSON.forEach((obj) => {
      numExamples += obj['num_examples'];
    });
    const n = tf.scalar(numExamples);
    updatesJSON.forEach((obj) => {
      const nk = tf.scalar(obj['num_examples']);
      obj['variables'].forEach(
          (variable: {values: number[], shape: number[]}, i: number) => {
            const tensor = tf.tensor(variable.values, variable.shape);
            updatedVars[i] = updatedVars[i].add(tensor.mul(nk.div(n)));
          });
    });
    const newModelId = new Date().getTime().toString();
    const newModelDir = path.join(dataDir, newModelId);
    const newModelPath = path.join(dataDir, newModelId + '.json');
    const newModelJSON = JSON.stringify({
      'variables': updatedVars.map((variable) => {
        return {'values': variable.dataSync(), 'rank': variable.shape};
      })
    });
    fs.writeFileSync(newModelPath, newModelJSON);
    fs.mkdirSync(newModelDir);
    if (modelId === currentModelId) {
      currentModelId = newModelId;
    } else {
      console.log('Another update beat us to the punch!');
    }
  }
}

app.get('/', (req: Request, res: Response) => {
  res.sendFile(indexPath);
});

app.post('/updates/:modelId', (req: Request, res: Response) => {
  const modelId = req.param('modelId');
  const varVals = req.param('variables');
  const updateId = uuid();
  const updatePath = path.join(dataDir, modelId, updateId);
  fs.writeFileSync(updatePath, varVals);
  if (modelId === currentModelId) {
    res.sendStatus(200);
    possiblyUpdateModel(modelId);
  } else {
    res.sendStatus(400);
  }
});

io.on('connection', (socket: socketIO.Socket) => {
  console.log('a user connected');
});

server.listen(3000, () => {
  console.log('listening on 3000');
});
