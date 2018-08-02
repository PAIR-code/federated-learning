Federated learning experiment using TensorFlow.js

## Basic Usage

On the server side:

```js
import * as express from 'express';
import * as http from 'http';
import * as federated from 'tfjs-federated-learning-server';

const expressApp = express();
const httpServer = http.createServer(expressApp);
const server = federated.Server(httpServer, 'https://initial.com/model.json');

server.setup().then(() => {
  httpServer.listen(PORT);
})
```

On the client side:

```js
import * as federated from 'tfjs-federated-learning-client';

const client = federated.Client(SERVER_URL, 'https://initial.com/model.json');

client.setup().then(() => {
  // make predictions!
  const yhat = client.predict(x);

  // train!
  client.fit(x, y);

  // listen for updates!
  client.onUpdate(() => {
    console.log(client.modelVersion);
  });
});
```

## Advanced Usage

See separate [server](./src/server/README.md) and [client](./src/client/README.md) docs.

