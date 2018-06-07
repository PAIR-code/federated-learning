import * as tf from '@tensorflow/tfjs';
import * as express from 'express';
import {Request, Response} from 'express';
import * as http from 'http';
import * as path from 'path';
import * as socketIO from 'socket.io';

import {ConnectionMsg, Events, VarsMsg} from '../common';
import {deserializeVar, serializeVar} from '../serialization';

const app = express();
const server = http.createServer(app);
const io = socketIO(server);

app.get('/', (req: Request, res: Response) => {
  res.sendFile(path.resolve(__dirname + '/../demo/index.html'));
});

function getInitVars() {
  const one = tf.tensor1d([1]);
  const oneV = tf.variable(one, true, 'oneV');
  return [oneV];
}

const initVars = getInitVars();
let clients = 0;
io.on('connection', async (socket: socketIO.Socket) => {
  console.log('a user connected');
  const vars = await Promise.all(initVars.map(serializeVar));
  clients++;
  const clientId = clients.toString();
  const msg: ConnectionMsg =
      {clientId, fitConfig: {batchSize: 10}, modelId: '0-0', initVars: vars};
  socket.emit(Events.Initialise, msg);

  setTimeout(async () => {
    const three = tf.tensor1d([30]);
    initVars[0].assign(three);
    const vars = await Promise.all(initVars.map(serializeVar));

    const updateMsg: VarsMsg = {clientId, modelId: '0-1', vars};
    socket.emit(Events.Download, updateMsg);
  }, 2000);

  socket.on(Events.Upload, (varMsg: VarsMsg, ack) => {
    const vars = varMsg.vars.map(deserializeVar);
    console.log('recv:', vars.map(x => x.dataSync()));
    ack(true);
  });
});

io.on(Events.Upload, async (socket: socketIO.Socket) => {
  console.log('gotem');
});

server.listen(3000, () => {
  console.log('listening on 3000');
});
