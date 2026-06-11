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
const PYTHON_BIN = process.env.CONDA_PYTHON || process.env.PYTHON_BIN || 'python3';
const PREDICT_SCRIPT = path.resolve(__dirname, '..', 'python_api', 'predict.py');
const LIST_MODELS_SCRIPT = path.resolve(__dirname, '..', 'python_api', 'list_models.py');

function runPythonScript(scriptPath, input = null) {
  return new Promise((resolve, reject) => {
    const py = spawn(PYTHON_BIN, [scriptPath]);
    let out = '';
    let err = '';

    py.stdout.on('data', (data) => { out += data.toString(); });
    py.stderr.on('data', (data) => { err += data.toString(); });

    py.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(err || `python exited ${code}`));
      }
      try {
        resolve(JSON.parse(out));
      } catch (e) {
        reject(new Error(`Invalid JSON from python: ${out}`));
      }
    });

    if (input != null) {
      py.stdin.write(JSON.stringify(input));
    }
    py.stdin.end();
  });
}

app.get('/models', async (_req, res) => {
  try {
    const data = await runPythonScript(LIST_MODELS_SCRIPT);
    return res.json(data);
  } catch (e) {
    return res.status(500).json({ error: e.message || String(e) });
  }
});

app.post('/predict', async (req, res) => {
  try {
    const parsed = await runPythonScript(PREDICT_SCRIPT, req.body || {});
    return res.json(parsed);
  } catch (e) {
    return res.status(500).json({ error: e.message || String(e) });
  }
});

const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || '0.0.0.0';
app.listen(PORT, HOST, () => {
  console.log(`Bike model API listening on ${HOST}:${PORT}`);
});
