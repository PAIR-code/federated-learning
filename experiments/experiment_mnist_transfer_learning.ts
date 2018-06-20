/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

Error.stackTraceLimit = Infinity;

import '../src/server/fetch_polyfill';
import '@tensorflow/tfjs-node';

import * as tf from '@tensorflow/tfjs';
import {v4 as uuid} from 'uuid';

import {VariableSynchroniser} from '../src/client/comm';

import {loadMnist} from './mnist_data';
import {MnistTransferLearningModel} from './mnist_transfer_learning_model';

function zip<T, U>(x: T[], y: U[]) {
  return x.map((_, i) => [x[i], y[i]] as [T, U]);
}

function preprocess(img: tf.Tensor) {
  return tf.tidy(() => img.sub(tf.scalar(127.5)).div(tf.scalar(127.5)));
}

async function main(
    splitStart: number, splitEnd: number, syncEvery = 1, verbose = true) {
  const loggingClientName = uuid().split('-')[1];
  const log = verbose ?
      (...args: any[]) => console.log(loggingClientName, ...args, Date.now()) :
      () => {};

  let done = false;

  const localExamples = splitEnd - splitStart;
  const fedModel = new MnistTransferLearningModel();
  const {vars, loss} = await fedModel.setup();
  let updateIdx = 0;
  const evaluate = () =>
      tf.tidy(() => loss(valImgs, tf.oneHot(valLabels, 10).toFloat()).mean())
          .dataSync()
          .slice();
  const sync = new VariableSynchroniser(vars, () => {
    updateIdx++;
    log('update recv', updateIdx);
    if (done) {
      const evalRes = evaluate();
      log('post update', updateIdx, 'sync loss', evalRes[0],
          'init loss:', preEvalRes[0]);
    }
    return true;
  });
  const fitConfig = await sync.initialise('http://localhost:3000');
  log('connected to server', 'processing split', splitStart, splitEnd);
  console.log(fitConfig);
  const batchSize = fitConfig.batchSize;

  const {train: {imgs: allImgs, labels: allLabels}, val: valData} = loadMnist();

  const {imgs: allValImgs, labels: allValLabels} = valData;

  const split = <T extends tf.Tensor>(t: T) =>
      tf.slice(t, [splitStart], [splitEnd - splitStart]);

  const [imgs, labels] = [split(allImgs), split(allLabels)];
  const [valImgs, valLabels] = [
    tf.slice(allValImgs, [splitStart], [512]),
    tf.slice(allValLabels, [splitStart], [512])
  ];

  if ((localExamples % batchSize) !== 0) {
    throw new Error(`${
        // tslint:disable-next-line:max-line-length
        loggingClientName} local batchsize must exactly divide number of local examples`);
  }

  const imgsBatches = tf.split(imgs, localExamples / batchSize).map(preprocess);
  const labelsBatches = tf.split(labels, localExamples / batchSize);

  allImgs.dispose();
  allLabels.dispose();
  imgs.dispose();
  labels.dispose();
  allValImgs.dispose();
  allValLabels.dispose();

  const preEvalRes = evaluate();
  log('initial loss', preEvalRes[0]);

  let i = 0;
  let wait = 100 + 50 * Math.random();
  // so all the clients don't try and sync at once
  let j = i + Math.floor(Math.random() * syncEvery);
  const optimizer = tf.train.sgd(0.001);
  for (const [img, label] of zip(imgsBatches, labelsBatches)) {
    optimizer.minimize(() => loss(img, tf.oneHot(label, 10).toFloat()));
    sync.numExamples += batchSize;
    i++;
    j++;

    if (j % syncEvery) {
      continue;
    }
    await new Promise((res, rej) => setTimeout(res(), wait));

    try {
      await sync.uploadVars();

      wait = 100 + 50 * Math.random();
      log('up sync', i, 'batch loss',
          loss(img, tf.oneHot(label, 10).toFloat()).mean().dataSync()[0]);
    } catch (exn) {
      wait = wait * 2.0;  // exp backoff
      j--;                // try again next iter
      log('timeout', exn);
    }
  }
  // process any pending updates
  await new Promise((res, rej) => setTimeout(res(), 50));
  log('done, evaluating final loss');
  done = true;
  const evalRes = evaluate();
  log('final loss', evalRes[0], 'init loss:', preEvalRes[0]);
  // sync.dispose();
  return;
}

(async () => {
  try {
    await main(
        parseInt(process.argv[2], 10),
        parseInt(process.argv[2], 10) + parseInt(process.argv[3], 10),
        parseInt(process.argv[4], 10));
  } catch (exn) {
    console.error(exn);
  }
})();
