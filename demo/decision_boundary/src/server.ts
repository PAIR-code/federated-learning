import '@tensorflow/tfjs-node';

import * as tf from '@tensorflow/tfjs';
import * as federated from 'federated-learning-server';
import * as fs from 'fs';
import * as http from 'http';

const httpServer = http.createServer();

const MODEL_DIR = './models';

if (!fs.existsSync('./models')) {
  fs.mkdirSync(MODEL_DIR)
}

function setupModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({inputShape: [2], units: 4, activation: 'relu'}),
      tf.layers.dense({units: 2, activation: 'relu'}),
    ]
  })
  return model
}

async function main() {
  const model = setupModel();
  const server = new federated.Server(httpServer, model, {
    clientHyperparams: {learningRate: 3e-4},
    updatesPerVersion: 2,
    modelDir: './models'
  });

  await server.setup();
  httpServer.listen(3000, () => {
    console.log('listening on 3000');
  })
}

main();
