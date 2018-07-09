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
import * as fileUpload from 'express-fileupload';
import * as federatedServer from 'federated-learning-server';
import * as fs from 'fs';
import * as http from 'http';
import * as path from 'path';
import * as io from 'socket.io';
import * as uuid from 'uuid/v4';

import {labelNames, loadAudioTransferLearningModel} from './model';

const dataDir = path.resolve(__dirname + '/data');
const fileDir = path.join(dataDir, 'files');
const mkdir = (dir) => !fs.existsSync(dir) && fs.mkdirSync(dir);

const app = express();
const httpServer = http.createServer(app);
const sockServer = io(httpServer);
const port = process.env.PORT || 3000;

app.use(fileUpload());

// tslint:disable-next-line:no-any
app.use((req: any, res: any, next: any) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  next();
});

// tslint:disable-next-line:no-any
app.post('/data', (req: any, res: any) => {
  if (!req.files) {
    return res.status(400).send('Must upload a file');
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
});

loadAudioTransferLearningModel().then(model => {
  federatedServer.setup(sockServer, model, dataDir).then(() => {
    mkdir(fileDir);
    for (let i = 0; i < labelNames.length; i++) {
      mkdir(path.join(fileDir, labelNames[i]));
    }

    httpServer.listen(port, () => {
      console.log(`listening on ${port}`);
    });
  });
});
