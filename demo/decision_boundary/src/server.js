const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const federated = require('federated-learning-server');
const http = require('http');

const httpServer = http.createServer();

function setupModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 10, inputShape: [2], activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 10, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 2, activation: 'softmax'}));
  return model;
}

async function main() {
  const model = setupModel();
  const server = new federated.Server(httpServer, model, {
    clientHyperparams: {learningRate: 1e-3, examplesPerUpdate: 2},
    serverHyperparams: {minUpdatesPerVersion: 2},
    modelDir: './models',
    verbose: true
  });

  await server.setup();
  httpServer.listen(3000, () => {
    console.log('listening on 3000');
  });
}

main();
