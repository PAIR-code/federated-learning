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

import * as express from 'express';
import * as fileUpload from 'express-fileupload';
import * as federatedServer from 'federated-learning-server';
import * as fs from 'fs';
import * as http from 'http';
import * as https from 'https';
import {resolve} from 'path';
import * as io from 'socket.io';
import {verify} from './verify';
import {setupModel} from './model';
import * as cookieParser from 'cookie-parser';
federatedServer.verbose(true);

const dataDir = resolve(__dirname + '/modelData');
const fileDir = resolve(__dirname + '/trainingData');

const mkdir = (dir: string) => !fs.existsSync(dir) && fs.mkdirSync(dir);
mkdir(fileDir)

const app = express();

let port: number;
let httpServer: http.Server | https.Server;
if (process.env.SSL_KEY && process.env.SSL_CERT) {
  const httpsOptions = {
    key: fs.readFileSync(process.env.SSL_KEY),
    cert: fs.readFileSync(process.env.SSL_CERT)
  };
  httpServer = https.createServer(httpsOptions, app);
  port = parseInt(process.env.PORT, 10) || 443;
} else {
  httpServer = http.createServer(app);
  port = parseInt(process.env.PORT, 10) || 3000;
}

const sockServer = io(httpServer);

app.use(express.static(resolve(`${__dirname}/../client/dist`)));

app.use(cookieParser());

if(process.env.USE_OAUTH) {
  app.use(async (req, res, next) => {
    try {
      const token = req.cookies['oauth2token'];
      const userid = await verify(token);
      (req as any).googleUserID = userid;
      next();
    } catch (exn) {
      console.log('unauthorized connection: ', exn.message);
      res.status(403).send({err: exn.message}).end();
    }
  });
}

app.use(fileUpload());

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', '*');
  next();
});

app.post('/data', (req, res) => {
  if (req.files == null) {
    return res.status(400).send('Must upload a file');
  }

  const file = req.files.file as fileUpload.UploadedFile;
  return file.mv(
      `${fileDir}/${file.name}`,
      err => {err ? res.status(500).send(err.toString()) :
                    res.send('Uploaded!')});
});

setupModel().then(({varsAndLoss}) => {
  federatedServer.setup(sockServer, varsAndLoss, dataDir, 1).then((server) => {
    server.setHyperparams({'learningRate': 3e-4});
    mkdir(fileDir);
    httpServer.listen(port, () => console.log(`listening on ${port}`));
  });
});
