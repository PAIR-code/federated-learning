require('@tensorflow/tfjs-node');
const federated = require('federated-learning-server');
const http = require('http');
const model = require('./model');

const httpServer = http.createServer();

async function main() {
  const server = new federated.Server(httpServer, model, {
    clientHyperparams: {learningRate: 1e-3, examplesPerUpdate: 5, epochs: 10},
    serverHyperparams: {minUpdatesPerVersion: 5},
    modelDir: './models',
    verbose: true
  });

  await server.setup();
  httpServer.listen(3000, () => {
    console.log('listening on 3000');
  });
}

main();
