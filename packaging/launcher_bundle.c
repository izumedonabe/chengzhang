/* 成章.app launcher for the self-contained build (tools/build_dmg.py).
 *
 * Hosts the bundled Python (Contents/Resources/python) in this process, so the
 * app is 成章 to macOS — Dock, menu bar, privacy permissions — and needs
 * nothing installed on the user's machine. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    char exe[PATH_MAX], real[PATH_MAX], home[PATH_MAX], python[PATH_MAX];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) != 0 || !realpath(exe, real)) {
        fprintf(stderr, "成章: cannot locate the app bundle\n");
        return 1;
    }
    char *macos = dirname(real);  /* …/成章.app/Contents/MacOS */
    snprintf(home, sizeof(home), "%s/../Resources/python", macos);
    snprintf(python, sizeof(python), "%s/bin/python3.12", home);

    PyStatus status;

    /* Apps opened from Finder get no LANG/LC_* at all; without UTF-8 mode Python
       would fall back to ASCII and choke on the first non-ASCII file it reads. */
    PyPreConfig preconfig;
    PyPreConfig_InitIsolatedConfig(&preconfig);
    preconfig.utf8_mode = 1;
    status = Py_PreInitialize(&preconfig);
    if (PyStatus_Exception(status)) Py_ExitStatusException(status);

    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);  /* ignore PYTHON* env vars and user site-packages */
    config.isolated = 1;
    config.write_bytecode = 0;             /* the bundle is signed; don't write into it */

    status = PyConfig_SetBytesString(&config, &config.home, home);
    if (PyStatus_Exception(status)) goto fail;
    status = PyConfig_SetBytesString(&config, &config.executable, python);
    if (PyStatus_Exception(status)) goto fail;
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
