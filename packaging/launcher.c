/* 成章.app launcher.
 *
 * Runs Python *inside this process* (instead of exec'ing the python binary),
 * so macOS sees the app bundle: Dock icon, menu-bar name, and microphone /
 * accessibility permissions all belong to 成章, not to "python3.12".
 *
 * VENV_PYTHON is baked in at build time (tools/build_app.py). */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifndef VENV_PYTHON
#error "build with -DVENV_PYTHON=..."
#endif

int main(int argc, char **argv) {
    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    /* Pretend to be the venv's interpreter: getpath then finds pyvenv.cfg and
       puts the venv's site-packages (MLX, pywebview, …) on sys.path. */
    status = PyConfig_SetBytesString(&config, &config.executable, VENV_PYTHON);
    if (PyStatus_Exception(status)) goto fail;
    status = PyConfig_SetBytesString(&config, &config.program_name, VENV_PYTHON);
    if (PyStatus_Exception(status)) goto fail;

    /* python -m chengzhang.app, plus any extra arguments */
    status = PyConfig_SetBytesString(&config, &config.run_module, "chengzhang.app");
    if (PyStatus_Exception(status)) goto fail;
    status = PyConfig_SetBytesArgv(&config, argc, argv);
    if (PyStatus_Exception(status)) goto fail;
    config.parse_argv = 0;

    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) goto fail;
    PyConfig_Clear(&config);
    return Py_RunMain();

fail:
    PyConfig_Clear(&config);
    Py_ExitStatusException(status);
}
