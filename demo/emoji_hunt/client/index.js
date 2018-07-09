import * as tf from '@tensorflow/tfjs'

console.log(tf);

import {SCAVENGER_HUNT_LABELS} from './labels.js';

`
import * as emojiName from '//unpkg.com/emoji-name-map?module';

import * as tf from '//unpkg.com/@tensorflow/tfjs/dist/tf.esm.js?module'
import {SCAVENGER_HUNT_LABELS} from './labels.js';
`


import * as ui from './ui.js';
import {EMOJIS_LVL_1} from './levels2.js';

const MODEL_URL = 'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/web_model.pb';
const WEIGHT_MANIFEST = 'https://storage.googleapis.com/learnjs-data/emoji_scavenger_hunt/weights_manifest.json'
const T_127_5 = tf.scalar(255 / 2);
const SERVER_URL = 'http://localhost:3000';

const levels = [EMOJIS_LVL_1];

for(const level of levels) {
  for(const {name, emoji} of level) {
    //emojiName.emoji[name] = emoji;
  }
}


async function loadModel() {
  const model = await tf.loadFrozenModel(MODEL_URL,  WEIGHT_MANIFEST);

 /* doesnt work for frozen model
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

async function main() {
  ui.status('loading model...');

  const model = await loadModel();
  // const client = new ClientAPI(model);

  // client.onDownload(msg => ui.modelVersion(msg.modelVersion));

  ui.status('trying to connect to federated learning server...');

 // await client.connect(SERVER_URL);

  ui.status('trying to get access to webcam...');

  const webcam = await ui.webcam();

  while(webcam.videoHeight === 0) {
    ui.status('waiting for video to initialise...')
    await tf.nextFrame();
  }

  let trainingMode = false;

  ui.overrideButton().addEventListener('click',  evt => {
    if(trainingMode) return;
    ui.status('ok! training now...')
    trainingMode = true;
    setTimeout(() => {
      trainingMode = false;
      ui.status('ready!');
    }, 2000)
  });

  ui.status('ready!');

  const pick = () => EMOJIS_LVL_1[Math.floor(EMOJIS_LVL_1.length * Math.random())]
  let lookingFor = pick();
  ui.findMe(`find me a ${lookingFor.name}, ${lookingFor.emoji}`)

  while(true) {
    await tf.nextFrame();
    if(trainingMode) continue;
    const preds = tf.tidy(() => {
      const frame = tf.fromPixels(webcam).toFloat();
      const scaled = tf.image.resizeBilinear(frame, [224, 224]);
      const prepped = scaled.sub(T_127_5).div(T_127_5).expandDims(0);
      return model.predict(prepped)
    });
    const { index, label, emoji } = await getTopPred(preds);

    ui.status(`its a ${label}${emoji ? `, ${emoji}!` : '!'}`);
    if(label == lookingFor.name) {
      ui.status(`congrats! u did it !`);
      for(let i = 0; i < 30; i++) {
        await tf.nextFrame();
      }
      lookingFor = pick();
      ui.findMe(`find me a ${lookingFor.name}, ${lookingFor.emoji}`)
    }
  }
}

main();
