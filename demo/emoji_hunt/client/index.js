import * as tf from '@tensorflow/tfjs'

import { FederatedDynamicModel, ClientAPI } from 'federated-learning-client';

import * as ui from './ui.js';

import {SCAVENGER_HUNT_LABELS} from './labels.js';
import {EMOJIS_LVL_1} from './levels2.js';

const MODEL_URL = 'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/web_model.pb';
const WEIGHT_MANIFEST = 'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/weights_manifest.json'

console.log(FederatedDynamicModel, ClientAPI);

const T_127_5 = tf.scalar(255 / 2);

const SERVER_URL = 'http://localhost:3000';

// Load the model & set it up for training
async function setupModel() {
  const model = await tf.loadFrozenModel(MODEL_URL,  WEIGHT_MANIFEST);
  const vars = model.executor.weightMap;

  // TODO: there must be a better way
  const nonTrainables = /(batchnorm)|(reshape)/g;

  // Make weights trainable & extract them
  const trainable = []

  for(const weightName in vars) {
    if(!weightName.match(nonTrainables)) {
      vars[weightName] = vars[weightName].map(t => {
        if(t.dtype === 'float32') {
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
  const loss = (input, label) => {
    const preds = model.predict(input);
    return tf.losses.logLoss(label, preds);
    }

  const optimizer = tf.train.sgd(0.1);

  const varsAndLoss = new FederatedDynamicModel(trainable, loss, optimizer)
  return { model, varsAndLoss };
}

async function getTopPred(preds) {
  const idx = tf.tidy(() => preds.argMax(1));
  const data = await idx.data();
  idx.dispose();
  const top = data[0];
  return { index: top, label: SCAVENGER_HUNT_LABELS[top] }
}

function preprocess(webcam) {
  return tf.tidy(() => {
    const frame = tf.fromPixels(webcam).toFloat();
    const scaled = tf.image.resizeBilinear(frame, [224, 224]);
    const prepped = scaled.sub(T_127_5).div(T_127_5).expandDims(0);
    return prepped
  });
}

async function main() {
  ui.status('loading model...');

  const { model, varsAndLoss } = await setupModel();

  const client = new ClientAPI(varsAndLoss);

  client.onDownload(msg => {
    console.log(msg);
    ui.modelVersion(`model version: ${msg.modelVersion}`);
  });

  ui.status('trying to connect to federated learning server...');

  await client.connect(SERVER_URL);

  ui.status('trying to get access to webcam...');

  const webcam = await ui.webcam();

  while(webcam.videoHeight === 0) {
    ui.status('waiting for video to initialise...');
    await tf.nextFrame();
  }

  let isTraining = false;

  ui.overrideButton(async evt => {
    if(isTraining) return;
    ui.status('ok! training now...');
    isTraining = true;
  });

  ui.status('ready!');


  const numLabels = Object.keys(SCAVENGER_HUNT_LABELS).reduce((x, y) => Math.max(x, y)) + 1
  const pickTarget = () => {
    const idx = Math.floor(EMOJIS_LVL_1.length * Math.random());
    const { name, emoji, path } = EMOJIS_LVL_1[idx];
    const targetIdx = Object.entries(SCAVENGER_HUNT_LABELS).filter(([idx, val]) => val == name)[0][0]
    return { name, emoji, path, targetIdx: parseInt(targetIdx) }
  }

  let lookingFor = pickTarget();
  ui.findMe(`find me a ${lookingFor.name}, ${lookingFor.emoji}`);

  while(true) {
    await tf.nextFrame();
    if(isTraining) {
        const [input, label] = tf.tidy(() => {
          const input = preprocess(webcam);
          const label = tf.oneHot([lookingFor.targetIdx], numLabels).toFloat();
          return [input, label];
        });

        await client.federatedUpdate(input, label);

        input.dispose();
        label.dispose();

        isTraining = false;
    };

    const preds = tf.tidy(() => {
      return model.predict(preprocess(webcam));
    });

    const { label } = await getTopPred(preds);

    preds.dispose();

    ui.status(`its a ${label}`);
    if(label == lookingFor.name) {
      ui.status(`congrats! u did it !`);
      for(let i = 0; i < 30; i++) {
        await tf.nextFrame();
      }
      lookingFor = pickTarget();
      ui.findMe(`find me a ${lookingFor.name}, ${lookingFor.emoji}`)
    }
  }
}

main();
