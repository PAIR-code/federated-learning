import * as tf from '@tensorflow/tfjs';
import * as federated from 'federated-learning-client';
import * as ui from './ui';

const SERVER_URL = 'http://localhost:3000';

function setupModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 10, inputShape: [2], activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 10, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 2, activation: 'softmax'}));
  return model;
}

async function main() {
  const client = new federated.Client(SERVER_URL, setupModel, {
    verbose: true
  });
  await client.setup();

  ui.setupUI();
  ui.syncClient(client);
  client.onNewVersion(() => {
    ui.syncClient(client);
  })

  ui.onClick.push(({label, x, y}) => {
    const xNorm = x / ui.CANVAS_SIZE;
    const yNorm = y / ui.CANVAS_SIZE;
    const inputTensor = tf.tensor2d([[xNorm, yNorm]], [1, 2]);
    const labelTensor = tf.oneHot([label], 2);
    inputTensor.print();
    labelTensor.print();
    client.federatedUpdate(inputTensor, labelTensor);
  })
}

main();
