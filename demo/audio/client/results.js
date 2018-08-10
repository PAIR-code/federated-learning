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

async function setup() {
  const metricsRes = await fetch(serverURL + '/metrics', fetchOptions);
  const metrics = await metricsRes.json();
  const versions = Object.keys(metrics);
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

    const validationLine = {
      x: [],
      y: [],
      name: 'Validation',
      mode: 'lines+markers',
      type: 'scatter',
      marker: { size: 20 },
      line: {
        color: 'rgba(255,0,0,1)',
        width: 5,
      }
    }

    const clientScatters = {};
    let maxY = 0;

    for (let i = 0; i < versions.length; i++) {
      const v = versions[i];
      const clients = Object.keys(metrics[v].clients);
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

        const dataValues = metrics[v].clients[c];
        let clientMean = 0;
        dataValues.forEach(d => clientMean += d[lossFn] / dataValues.length);
        clientScatters[c].x.push(i+1);
        clientScatters[c].y.push(clientMean);
        clientScatters[c].text.push(`${dataValues.length} examples`);
        versionMean += clientMean / clients.length;
        maxY = Math.max(maxY, clientMean);
      }
      meanClientLine.x.push(i+1);
      meanClientLine.y.push(versionMean);
      meanClientLine.text.push(`${clients.length} clients`);

      if (metrics[v].validation) {
        const val = metrics[v].validation[lossFn];
        validationLine.x.push(i+1);
        validationLine.y.push(val);
        maxY = Math.max(maxY, val);
      }
    }

    const clients = Object.keys(clientScatters);
    const plots = clients.map(c => clientScatters[c]).concat(
      [validationLine, meanClientLine]);

    Plotly.newPlot('results', plots, {
      autosize: true,
      title: 'Federated Learning Progress',
      yaxis: {
        title: ['Cross-Entropy', 'Accuracy'][lossFn],
        range: [0, [maxY * 1.05, 1][lossFn]]
      },
      xaxis: {
        title: 'Model Version',
        range: [0, versions.length + 1]
      }
    });
  }

  const metric = document.getElementById('performance-metric');
  metric.onchange = () => {
    redrawGraph(parseInt(metric.value));
  }
  redrawGraph(parseInt(metric.value));
};

setup();
