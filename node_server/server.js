const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
app.use(bodyParser.json());
// Serve static frontend from public/
app.use(express.static(path.join(__dirname, 'public')));
// Serve generated chart images from outputs/
app.use('/outputs', express.static(path.resolve(__dirname, '..', 'outputs')));

// Path to Python executable inside conda env. Can be overridden with CONDA_PYTHON env var.
const PYTHON_BIN = process.env.CONDA_PYTHON || '/Users/double0/miniconda/envs/bike-env/bin/python';
const PREDICT_SCRIPT = path.resolve(__dirname, '..', 'python_api', 'predict.py');

app.post('/predict', async (req, res) => {
  const input = req.body || {};
  const py = spawn(PYTHON_BIN, [PREDICT_SCRIPT]);

  let out = '';
  let err = '';

  py.stdout.on('data', (data) => { out += data.toString(); });
  py.stderr.on('data', (data) => { err += data.toString(); });

  py.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ error: err || `python exited ${code}` });
    }
    try {
      const parsed = JSON.parse(out);
      return res.json(parsed);
    } catch (e) {
      return res.status(500).json({ error: 'Invalid JSON from python', raw: out, err });
    }
  });

  // send input as JSON via stdin
  py.stdin.write(JSON.stringify(input));
  py.stdin.end();
});

const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || '0.0.0.0';
app.listen(PORT, HOST, () => {
  console.log(`Bike model API listening on ${HOST}:${PORT}`);
});
