module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          PYTHONUNBUFFERED: "1"
        },
        path: "app",
        message: [
          "python app.py"
        ],
        on: [{
          // Gradio prints "Running on local URL:  http://127.0.0.1:<port>"
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      // Captured URL -> "Open Web UI" menu item in pinokio.js
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
