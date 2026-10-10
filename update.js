module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    // Backend environments show "outdated" in the Environments tab when their
    // pinned requirements change; update them there.
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: [
        "uv pip install -r requirements.txt",
        "uv pip install -e ./protocol"
      ]
    }
  }]
}
