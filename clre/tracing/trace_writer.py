#!/usr/bin/env python
"""Buffered npz trace writer (schema v1, docs/native_vad_36route_trace_schema.md).

In-RAM per-key lists; atomic partial flush every FLUSH_EVERY ticks
(tmp + os.replace) so a killed run loses at most the unflushed tail.
"""

import json
import os
import time

import numpy as np

FLUSH_EVERY = 1000


class TraceWriter:
    def __init__(self, out_dir, name):
        self.out_dir = out_dir
        self.path = os.path.join(out_dir, name + ".npz")
        os.makedirs(out_dir, exist_ok=True)
        self.buf = {}
        self.n = 0

    def append(self, **kv):
        for k, v in kv.items():
            self.buf.setdefault(k, []).append(v)
        self.n += 1
        if self.n % FLUSH_EVERY == 0:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        arrays = {}
        for k, lst in self.buf.items():
            try:
                arrays[k] = np.stack(lst) if isinstance(lst[0], np.ndarray) \
                    else np.asarray(lst)
            except Exception:
                arrays[k] = np.asarray(lst, dtype=object)
        tmp = self.path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, self.path)

    def close(self, meta_path=None, meta=None):
        self.flush()
        if meta_path and meta is not None:
            meta = dict(meta)
            meta["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            meta["n_ticks"] = self.n
            tmp = meta_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(meta, f, indent=2)
            os.replace(tmp, meta_path)
