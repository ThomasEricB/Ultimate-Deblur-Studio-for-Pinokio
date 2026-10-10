module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    // Master (control plane) venv on a current Python. Pinokio's `venv`
    // attribute activates an existing venv, so create it with uv first.
    {
      when: "{{!exists('app/env')}}",
      method: "shell.run",
      params: {
        path: "app",
        message: "uv venv env --python 3.13"
      }
    },
    // The master only needs Gradio + the shared worker protocol. Model
    // backends (NCNN Vulkan, PyTorch/spandrel, ...) are installed per backend,
    // in their own isolated venvs, from the app's Environments tab.
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "uv pip install -e ./protocol"
        ]
      }
    }
  ]
}
