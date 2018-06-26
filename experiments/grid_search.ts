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
