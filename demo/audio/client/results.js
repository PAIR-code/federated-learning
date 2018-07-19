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

let serverURL = location.origin;
let user = '';
let pass = '';
const fetchOptions = {};

if (URLSearchParams) {
  const params = new URLSearchParams(location.search);

  // Allow custom server URL
  if (params.get('server')) {
    serverURL = params.get('server');
  }

  // Slight hack to enable access to /data endpoint
  user = params.get('user');
  pass = params.get('pass');
  if (user && pass) {
    fetchOptions.headers = {
      'Authorization': 'Basic ' + btoa(user + ':' + pass)
    }
  };
}

const lossFns = {
  'Cross-Entropy': (d) => {
    const probs = d.modelOutput.split(',').map(parseFloat);
    return -Math.log(probs[parseInt(d.trueLabel)]);
  },
  'Accuracy': (d) => {
    return d.predictedLabel === d.trueLabel ? 1 : 0;
  }
}

async function setup() {
  const dataRes = await fetch(serverURL + '/data', fetchOptions);
  const data = await dataRes.json();
  const dataByVersionByClient = {};

  for (let i = 0; i < data.length; i++) {
    const version = data[i].modelVersion;
    const client = data[i].clientId
    if (!dataByVersionByClient[version]) {
      dataByVersionByClient[version] = {};
    }
    if (!dataByVersionByClient[version][client]) {
      dataByVersionByClient[version][client] = [];
    }
    dataByVersionByClient[version][client].push(data[i]);
  }

  const versions = Object.keys(dataByVersionByClient);
  versions.sort();

  function redrawGraph(lossFn) {
    const meanClientLine = {
      x: [],
      y: [],
      text: [],
      name: 'Avg. Client',
      mode: 'lines+markers',
      type: 'scatter',
      marker: { size: 20 },
      line: {
        color: 'rgba(0,0,255,1)',
        width: 5,
        dash: 'dashdot'
      }
    }

    const clientScatters = {};

    for (let i = 0; i < versions.length; i++) {
      const v = versions[i];
      const clients = Object.keys(dataByVersionByClient[v]);
      let versionMean = 0;

      for (let j = 0; j < clients.length; j++) {
        const c = clients[j];

        if (!clientScatters[c]) {
          clientScatters[c] = {
            x: [],
            y: [],
            text: [],
            name: `Client ${Object.keys(clientScatters).length+1}`,
            mode: 'lines+markers',
            type: 'scatter',
            opacity: 0.5,
            marker: {
              size: 12
            },
            line: {
              dash: 'dot'
            }
          }
        }

        const dataValues = dataByVersionByClient[v][c];
        let clientMean = 0;
        dataValues.forEach(d => {
          clientMean += lossFns[lossFn](d) / dataValues.length;
        });
        clientScatters[c].x.push(i+1);
        clientScatters[c].y.push(clientMean);
        clientScatters[c].text.push(`${dataValues.length} examples`);
        versionMean += clientMean / clients.length;
      }
      meanClientLine.x.push(i+1);
      meanClientLine.y.push(versionMean);
      meanClientLine.text.push(`${clients.length} clients`);
    }

    const clients = Object.keys(clientScatters);
    const plots = clients.map(c => clientScatters[c]).concat([meanClientLine]);

    Plotly.newPlot('results', plots, {
      autosize: true,
      title: 'Federated Learning Progress',
      yaxis: {
        title: lossFn
      },
      xaxis: {
        title: 'Model Version'
      }
    });
  }

  const metric = document.getElementById('performance-metric');
  metric.onchange = () => {
    redrawGraph(metric.value);
  }
  redrawGraph(metric.value);
};

setup();
