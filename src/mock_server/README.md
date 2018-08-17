# Tensorflow.js Federated Learning Mock Server

This library serves as a browser-compatible mock of the [Node.js server library](../server).

## Usage

```js
// client-side javascript
import {MockServer as FederatedServer} from 'federated-learning-mock-server';
import {Client as FederatedClient} from 'federated-learning-client';

const model = await tf.loadModel('https://my.model.json');

// create a mock, in-memory only version of the federated server
const server = new FederatedServer(model);
await server.setup();

// create two clients and link them to the server via `newClientSocket`
const client1 = new FederatedClient(server.newClientSocket, model);
const client2 = new FederatedClient(server.newClientSocket, model);
await client1.setup();
await client2.setup();
```
