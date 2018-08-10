import * as tf from '@tensorflow/tfjs';
import * as federated from 'federated-learning-client';
import * as ui from './ui';

const SERVER_URL = 'http://localhost:3000';

function setupModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [2], units: 4, activation: 'relu' }),
      tf.layers.dense({ units: 2, activation: 'relu' }),
    ]
  })
  console.log(model);
  return model
}

async function main() {

  const client = new federated.Client(SERVER_URL, setupModel);

  await client.setup();
  ui.setupUI();

  ui.onClick.push(({label, x, y}) => {
    const xNorm = x / ui.CANVAS_SIZE;
    const yNorm = y / ui.CANVAS_SIZE;
    const inputTensor = tf.tensor2d([[xNorm, yNorm]], [1, 2]);
    const labelTensor = tf.oneHot([label], 2);
    client.federatedUpdate(inputTensor, labelTensor);
  })
}

main();
