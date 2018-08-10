import * as http from 'http';
import * as federated from 'federated-learning-server';

const httpServer = http.httpServer();

function setupModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [2], units: 4, activation: 'relu' }),
      tf.layers.dense({ units: 2, activation: 'relu' }),
    ]
  })
  return model
}

async function main() {
  const model = setupModel();
  const server = federated.Server(httpServer, model, {

  })
}
