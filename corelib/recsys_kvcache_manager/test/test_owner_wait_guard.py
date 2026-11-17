#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only regression: a held-GIL native caller waits for a GIL-using worker.

The unguarded child must hit its watchdog, while guarded waits and exception
unwinding complete. Also checks that the header works in a Python-free binary.
No Torch import, cache server, CUDA initialization, or GPU work is involved.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

SOURCE = r'''
#include <Python.h>
#include <future>
#include <thread>
#include "owner_wait_guard.h"
static PyObject* wait_once(PyObject*, PyObject* args) {
    int guarded, fail = 0;
    if (!PyArg_ParseTuple(args, "p|p", &guarded, &fail)) return nullptr;
    std::promise<void> promise;
    auto future = promise.get_future();
    std::thread worker([&] {
        // Models PyTorch TensorImpl's Python refcount hook on an owner thread.
        auto gil = PyGILState_Ensure();
        if (fail) promise.set_exception(std::make_exception_ptr(std::runtime_error("worker error")));
        else promise.set_value();
        PyGILState_Release(gil);
    });
    std::exception_ptr error;
    if (guarded) {
        kvcache_manager::OwnerWaitGuard guard;
        try { future.get(); } catch (...) { error = std::current_exception(); }
        worker.join();
    } else {
        future.wait();
        worker.join();
    }
    if (!PyGILState_Check()) Py_FatalError("owner wait failed to restore GIL");
    if (error) { PyErr_SetString(PyExc_RuntimeError, "worker error"); return nullptr; }
    Py_RETURN_TRUE;
}
static PyMethodDef methods[] = {
    {"wait_once", wait_once, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "owner_gil_test", nullptr, -1, methods};
PyMODINIT_FUNC PyInit_owner_gil_test() { return PyModule_Create(&module); }
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--header", type=Path, default=Path(__file__).resolve().parents[1] / "src/runtime/owner_wait_guard.h")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "owner_gil_test.cpp"
    source.write_text(SOURCE)
    module = output / ("owner_gil_test" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-pthread",
                    "-I" + sysconfig.get_paths()["include"], "-I" + str(args.header.resolve().parent),
                    str(source), "-ldl", "-o", str(module)], check=True)
    setup = f"import sys;sys.path.insert(0,{str(output)!r});import owner_gil_test as m;"
    negative = subprocess.Popen([sys.executable, "-c", setup + "print('entered',flush=True);m.wait_once(False)"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        negative.communicate(timeout=3)
        raise AssertionError("unguarded held-GIL wait unexpectedly completed")
    except subprocess.TimeoutExpired:
        negative.kill()
        stdout, stderr = negative.communicate()
        assert "entered" in stdout, (stdout, stderr)
    positive = setup + "\nfor _ in range(100): assert m.wait_once(True)\ntry: m.wait_once(True,True)\nexcept RuntimeError: pass\nelse: raise AssertionError('exception missing')\nprint('guarded-ok')"
    passed = subprocess.run([sys.executable, "-c", positive], check=True, timeout=10, capture_output=True, text=True)
    assert "guarded-ok" in passed.stdout
    native_source = output / "native_wait.cpp"
    native_source.write_text('#include "owner_wait_guard.h"\nint main(){kvcache_manager::OwnerWaitGuard guard;}\n')
    native = output / "native_wait"
    subprocess.run(["g++", "-std=c++17", "-I" + str(args.header.resolve().parent), str(native_source),
                    "-ldl", "-o", str(native)], check=True)
    subprocess.run([str(native)], check=True, timeout=3)
    result = {"unguarded_watchdog_reproduced": True, "guarded_waits": 100,
              "exception_restores_gil": True, "native_without_python": True}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
