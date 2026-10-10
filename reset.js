// Removes every installed environment (master + backend venvs) and runtime
// scratch files. Downloaded models, outputs and settings are kept; delete
// app/models or app/outputs by hand if you want those gone too.
module.exports = {
  run: [{
    method: "fs.rm",
    params: { path: "app/env" }
  }, {
    method: "fs.rm",
    params: { path: "app/plugins/ncnn-vulkan/.venv" }
  }, {
    method: "fs.rm",
    params: { path: "app/plugins/spandrel/.venv" }
  }, {
    method: "fs.rm",
    params: { path: "app/runtime" }
  }]
}
