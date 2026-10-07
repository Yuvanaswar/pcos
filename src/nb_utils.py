"""
Notebook generation and execution utilities.
Enables programmatic creation of compliant Jupyter v4 notebooks with live execution.
"""

import json
import os
import sys
import io
import traceback
import uuid
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

class TeeIO:
    def __init__(self, buffer, stream):
        self.buffer = buffer
        self.stream = stream

    def write(self, data):
        self.buffer.write(data)
        if self.stream:
            try:
                self.stream.write(data)
                self.stream.flush()
            except Exception:
                pass

    def flush(self):
        self.buffer.flush()
        if self.stream:
            try:
                self.stream.flush()
            except Exception:
                pass

    def getvalue(self):
        return self.buffer.getvalue()

class NotebookBuilder:
    def __init__(self, title: str = "Ovarian Ultrasound Benchmark"):
        self.cells = []
        self.execution_count = 0
        self.title = title

    def add_markdown(self, markdown_text: str):
        lines = [line + "\n" for line in markdown_text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        self.cells.append({
            "cell_type": "markdown",
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "source": lines
        })

    def add_code(self, code_text: str, outputs=None, execution_count=None):
        lines = [line + "\n" for line in code_text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        self.cells.append({
            "cell_type": "code",
            "execution_count": execution_count,
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "outputs": outputs if outputs is not None else [],
            "source": lines
        })

    def to_dict(self):
        return {
            "cells": self.cells,
            "metadata": {
                "language_info": {
                    "name": "python",
                    "version": "3.14"
                },
                "kernelspec": {
                    "display_name": "Python 3.14 (PCOS PyTorch)",
                    "language": "python",
                    "name": "python314"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

    def save(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        print(f"[NotebookBuilder] Saved notebook to: {filepath}")

    def execute_and_save(self, filepath: str, working_dir: str = None):
        """
        Sequentially executes all code cells in a shared namespace, capturing stdout
        and saving the executed outputs directly into the notebook JSON.
        """
        orig_cwd = os.getcwd()
        if working_dir:
            os.chdir(working_dir)

        global_scope = {"__name__": "__main__"}
        
        try:
            for cell in self.cells:
                if cell["cell_type"] == "code":
                    self.execution_count += 1
                    cell["execution_count"] = self.execution_count
                    code = "".join(cell["source"])
                    
                    stdout_buf = io.StringIO()
                    stderr_buf = io.StringIO()
                    tee_stdout = TeeIO(stdout_buf, sys.__stdout__)
                    tee_stderr = TeeIO(stderr_buf, sys.__stderr__)
                    
                    try:
                        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                            exec(code, global_scope)
                        
                        stdout_str = stdout_buf.getvalue()
                        stderr_str = stderr_buf.getvalue()
                        
                        outputs = []
                        if stdout_str:
                            outputs.append({
                                "name": "stdout",
                                "output_type": "stream",
                                "text": stdout_str.splitlines(keepends=True)
                            })
                        if stderr_str:
                            outputs.append({
                                "name": "stderr",
                                "output_type": "stream",
                                "text": stderr_str.splitlines(keepends=True)
                            })
                        cell["outputs"] = outputs
                    except Exception as e:
                        err_msg = traceback.format_exc()
                        cell["outputs"] = [{
                            "ename": type(e).__name__,
                            "evalue": str(e),
                            "output_type": "error",
                            "traceback": err_msg.splitlines(keepends=True)
                        }]
                        print(f"Error in cell {self.execution_count}: {e}")
                        raise e
        finally:
            os.chdir(orig_cwd)
            self.save(filepath)
