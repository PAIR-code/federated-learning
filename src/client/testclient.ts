import * as tf from '@tensorflow/tfjs';

import {VarsMsg} from '../common';

import {VariableSynchroniser} from './comm';
async function main() {
  const one = tf.tensor1d([1]);
  const oneV = tf.variable(one, true, 'oneV');
  const sync = new VariableSynchroniser([oneV], (varsMsg: VarsMsg) => {
    return true;
  });
  const interval = setInterval(() => {
    if (oneV.dataSync()[0] === 30) {
      console.log('downsync successful');
      clearInterval(interval);
    }
  });
  await sync.initialise('http://localhost:3000');
  console.log('initialised');
  await sync.uploadVars();
  console.log('first upload succeeded');
  oneV.assign(one.add(one));
  await sync.uploadVars();
  console.log('second upload succeeded');
}

main();
