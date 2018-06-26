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

import {spawn} from 'child_process';
import {createWriteStream} from 'fs';

const CLIENTS = [2, 11, 22, 33, 101, 1001];
const EXAMPLES = [5, 10, 30, 40];
const AVG_EVERY = [10, 100, 200];
const SYNC_EVERY = [1, 3, 5];

const f: any =
    (a: any[],
     b: any[]) => [].concat(...a.map(d => b.map(e => [].concat(d, e)))) as
    any[];
const cartesian: any = (a: any[], b?: any[], ...c: any[][]) =>
    (b ? cartesian(f(a, b), ...c) : a) as any[];

async function main() {
  for (const [nclients, nexamples, nsync, navg] of cartesian(
           CLIENTS, EXAMPLES, SYNC_EVERY, AVG_EVERY)) {
    if (nclients * nexamples < nsync || nclients * nexamples < navg ||
        nclients * nexamples + 512 > 10000) {
      continue;
    }

    const logFilename = `logs/${nclients}_${nexamples}_${nsync}_${navg}.txt`;
    const logf = createWriteStream(logFilename);
    const cmd = `/bin/bash ${__dirname}/reset_and_launch.sh ${nclients} ${
        nexamples} ${nsync} ${navg}`;
    console.error(
        'running', nclients, nexamples, nsync, navg, logFilename, cmd);
    const sink = spawn(
        'bash', ['./reset_and_launch.sh', nclients, nexamples, nsync, navg],
        {stdio: [process.stdin, 'pipe', process.stderr]});
    let lastOutput = Date.now();
    let anyClientEnded = false;
    let done: () => void;
    const doneProm = new Promise((res, rej) => {
      done = res;
    });
    const pollKillOnTimeout = setInterval(() => {
      if (anyClientEnded && Date.now() - lastOutput > 10000) {
        clearInterval(pollKillOnTimeout);
        sink.kill();
        done();
      }
    }, 500);
    sink.stdout.on('data', (chunk) => {
      if (!anyClientEnded) {
        const sbuf = chunk.toString();
        if (sbuf.indexOf('final loss') !== -1) {
          console.error('clients started to end');
          anyClientEnded = true;
        }
      }
      lastOutput = Date.now();
      logf.write(chunk);
    });
    await doneProm;
  }
}

main();
