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
import {loadFrozenModel} from '@tensorflow/tfjs-converter'
import * as federated from 'federated-learning-client';

import {SCAVENGER_HUNT_LABELS} from './labels.js';
import {EMOJIS_LVL_1} from './levels.js';
import {upload} from './training_data_upload.js';
import * as ui from './ui.js';

const MODEL_URL =
    'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/web_model.pb';
const WEIGHT_MANIFEST =
    'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/weights_manifest.json';

const SERVER_URL = `//${location.hostname}:3000`;
const UPLOAD_URL = `//${location.hostname}:3000/data`;
const USE_OAUTH = false;

console.log('server url:', SERVER_URL)

const MODEL_INPUT_WIDTH = 224;
const NUM_LABELS = 424;

const LEARNING_RATE = 0.1;

// Load the model & set it up for training
async function setupModel() {
  const model = await loadFrozenModel(MODEL_URL, WEIGHT_MANIFEST);
  const weights = model.weights;

  // TODO: there must be a better way
  const nonTrainables = /(batchnorm)|(reshape)/g;

  // Make weights trainable & extract them
  const vars = [];

  for (const weightName in weights) {
    if (!weightName.match(nonTrainables)) {
      weights[weightName] = weights[weightName].map(t => {
        if (t.dtype === 'float32') {
          const ret = tf.variable(t);
          vars.push(ret);
          return ret;
        } else {
          return t;
        }
      });
    }
  }

  // TODO: better to not run softmax and use softmaxCrossEntropy?
  const loss = (y, label) => tf.losses.logLoss(y, label)

  const inputShape = [MODEL_INPUT_WIDTH, MODEL_INPUT_WIDTH, 3];
  const outputShape = [NUM_LABELS];

  const varsAndLoss = new federated.FederatedDynamicModel({
    vars,
    predict: x => model.predict(x),
    loss,
    inputShape,
    outputShape
  });
  return varsAndLoss;
}

async function getTopPred(preds) {
  const idx = preds.argMax(1);
  const data = await idx.data();
  tf.dispose(idx);
  const top = data[0];
  return {index: top, label: SCAVENGER_HUNT_LABELS[top]};
}

// center-crop input Tensor3D into a square
function squareCrop(frame) {
  return tf.tidy(() => {
    const [h, w] = frame.shape;
    if(h > w) {
      const halfW = Math.floor(w / 2);
      const halfH = Math.floor(h / 2);
      return frame.slice([halfH - halfW, 0], [halfW * 2, -1]);
    } else {
      const halfH = Math.floor(h / 2);
      const halfW = Math.floor(w / 2);
      return frame.slice([0, halfW - halfH], [-1, halfH * 2]);
    }
  })
}

function preprocess(webcam) {
  return tf.tidy(() => {
    const frame = tf.fromPixels(webcam);
    const cropped = squareCrop(frame).toFloat();
    const scaled =
        tf.image.resizeBilinear(cropped, [MODEL_INPUT_WIDTH, MODEL_INPUT_WIDTH]);
    const prepped = scaled.sub(255 / 2).div(255 / 2).expandDims(0);
    return prepped;
  });
}

async function main() {
  const signedIn = (!USE_OAUTH) || await ui.login();
  if(!signedIn) {
    ui.status('please refresh & login to proceed');
    return;
  }

  ui.status('trying to get access to webcam...');

  const webcam = await ui.webcam();

  ui.status('loading model...');

  const model = await setupModel();

  const client = new federated.Client(SERVER_URL, model, {
    verbose: true
  });

  ui.modelVersion(`model version: ${client.modelVersion()}`);

  client.onNewVersion((oldVersion, newVersion) => {
    console.log(oldVersion, newVersion, client.modelVersion());
    ui.modelVersion(`model version: ${newVersion}`);
  });

  ui.status('trying to connect to federated learning server...');

  await client.setup();

  let isTraining = false;

  ui.overrideButton(evt => {
    if (isTraining) {
      return
    };
    ui.status('ok! training now...');
    isTraining = true;
  });

  ui.status('ready!');

  const numLabels =
      Object.keys(SCAVENGER_HUNT_LABELS).reduce((x, y) => Math.max(x, y)) + 1;

  const pickTarget = () => {
    const idx = Math.floor(EMOJIS_LVL_1.length * Math.random());
    const {name, emoji, path} = EMOJIS_LVL_1[idx];
    const [targetIdx, _] =
        Object.entries(SCAVENGER_HUNT_LABELS).filter(([idx,
                                                       val]) => val == name)[0];
    return {name, emoji, path, targetIdx: parseInt(targetIdx)};
  };

  let lookingFor = pickTarget();

  ui.findMe(`find me a ${lookingFor.name}, ${lookingFor.emoji}`);

  while (true) {
    await tf.nextFrame();

    if (isTraining) {
      const [input, label] = tf.tidy(() => {
        const input = preprocess(webcam);
        const label = tf.oneHot([lookingFor.targetIdx], numLabels).toFloat();
        return [input, label];
      });

      try {
        await client.federatedUpdate(input, label);

        if (ui.uploadAllowed()) {
          upload(UPLOAD_URL, lookingFor.targetIdx, webcam)
              .catch(err => ui.status(err));
        }

      } catch (err) {
        ui.status(err);
      }

      tf.dispose([input, label]);

      isTraining = false;

      lookingFor = pickTarget();
      ui.findMe(`find me a ${lookingFor.name}, ${lookingFor.emoji}`)
    };

    const preds = tf.tidy(() => {
      return client.predict(preprocess(webcam));
    });

    const {label} = await getTopPred(preds);

    tf.dispose(preds);

    ui.status(`i see a ${label}...`);
    if (label === lookingFor.name) {
      ui.status(`congrats! u did it !`);
      for (let i = 0; i < 30; i++) {
        await tf.nextFrame();
      }
      lookingFor = pickTarget();
      ui.findMe(`find me a ${lookingFor.name}, ${lookingFor.emoji}`)
    }
  }
}

main();
