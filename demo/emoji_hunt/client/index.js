import * as tf from '@tensorflow/tfjs-core'
import { loadFrozenModel } from '@tensorflow/tfjs-converter';

import {SCAVENGER_HUNT_LABELS} from './labels.js';
import * as ui from './ui.js';
import {EMOJIS_LVL_1} from './levels2.js';

const MODEL_URL = 'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/web_model.pb';
const WEIGHT_MANIFEST = 'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/weights_manifest.json'
const T_127_5 = tf.scalar(255 / 2);
const SERVER_URL = 'http://localhost:3000';

const levels = [EMOJIS_LVL_1];

async function loadModel() {
  const model = await loadFrozenModel(MODEL_URL,  WEIGHT_MANIFEST);
  const vars = model.executor.weightMap;
  const nonTrainables = /(batchnorm)|(reshape)/g;
  for(const weightName in vars) {
    if(!weightName.match(nonTrainables)) {
      vars[weightName] = vars[weightName].map(t => t.dtype === 'float32'
                                                   ? tf.variable(t, true)
                                                   : t);
    }
  }

 /*
  doesnt work for frozen model
  for(const layer of model.layers) {
    layer.trainable = false;
  }

  model.getLayer('conv_preds').trainable = true;
  */
  //model.compile({'optimizer': 'sgd', loss: 'categoricalCrossentropy'});

  return model;
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

  const model = await loadModel();

  //const client = new ClientAPI(model);

  client.onDownload(msg => ui.modelVersion(msg.modelVersion));

  ui.status('trying to connect to federated learning server...');

  //await client.connect(SERVER_URL);

  ui.status('trying to get access to webcam...');

  const webcam = await ui.webcam();

  while(webcam.videoHeight === 0) {
    ui.status('waiting for video to initialise...');
    await tf.nextFrame();
  }

  let isTraining = false;

  ui.overrideButton(evt => {
    if(isTraining) return;
    ui.status('ok! training now...');
    isTraining = true;
    setTimeout(() => {
      isTraining = false;
      ui.status('ready!');
    }, 2000);
  });

  ui.status('ready!');

  const pickTarget = () => {
    const idx = Math.floor(EMOJIS_LVL_1.length * Math.random());
    const { name, emoji, path } = EMOJIS_LVL_1[idx];
    const targetIdx = labels.indexOf(name);
    return { name, emoji, path, targetIdx }
  }

  let lookingFor = pickTarget();
  ui.findMe(`find me a ${lookingFor.name}, ${lookingFor.emoji}`);

  while(true) {
    await tf.nextFrame();
    if(isTraining) {
        const [x, y] = tf.tidy(() => {
          const input = preprocess(webcam);
          const targetIdxT = tf.scalar(lookingFor.targetIdx, 'int32');
          const labels = tf.oneHot(targetIdxT.expandDims(0));
          return [input, labels];
        });
        //await client.federatedUpdate(x, y);

        x.dispose();
        y.dispose();
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
