"""CFFI bindings to Kyse's opaque C API; no internal model structs are mirrored."""
from pathlib import Path
import sys
import threading
import weakref

from cffi import FFI

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT/'build'/('libkyse.dylib' if sys.platform == 'darwin' else 'libkyse.so')

# Keep this small ABI declaration in sync with src/kyse.h (API version 1).
DECLARATIONS = '''
typedef struct Kyse Kyse;
typedef struct { const char *data; size_t size; int id; } KyseToken;
typedef struct { int prompt_tokens, completion_tokens, length; } KyseUsage;
int kyse_api_version(void);
const char *kyse_error(void);
Kyse *kyse_open(const char *, const char *, const char *, int);
int kyse_close(Kyse *);
int kyse_begin(Kyse *, const char *, size_t, int, float, float, uint64_t);
int kyse_next(Kyse *, KyseToken *);
int kyse_stop(Kyse *);
int kyse_usage(Kyse *, KyseUsage *);
'''


class Model:
    """One native model per process. Use close() or a context manager for ownership."""
    def __init__(self, path, backend='cpu', context=4096, library=DEFAULT_LIBRARY, tokenizer=None):
        self._lock = threading.RLock()
        self._handle = None
        self.ffi = FFI()
        self.ffi.cdef(DECLARATIONS)
        self.lib = self.ffi.dlopen(str(Path(library).resolve()))
        if self.lib.kyse_api_version() != 1:
            raise RuntimeError('Unsupported Kyse shared-library API version; rebuild with make library.')
        self._token = self.ffi.new('KyseToken *')
        self._usage = self.ffi.new('KyseUsage *')
        handle = self.lib.kyse_open(str(Path(path).resolve()).encode(),
                                   str(Path(tokenizer).resolve()).encode() if tokenizer else self.ffi.NULL,
                                   backend.encode(), context)
        if handle == self.ffi.NULL:
            self._raise(-2)
        self._handle = handle
        self._finalizer = weakref.finalize(self, self.lib.kyse_close, handle)

    @property
    def closed(self):
        return self._handle is None

    def _raise(self, status):
        # Error text is thread-local, so copy it on the thread that called C.
        message = self.ffi.string(self.lib.kyse_error()).decode('utf-8', 'replace')
        raise (ValueError if status == -1 else RuntimeError)(message)

    def _check(self, status):
        if status < 0:
            self._raise(status)
        return status

    def _require_open(self):
        if self.closed:
            raise RuntimeError('Model is closed.')

    def begin(self, prompt, max_tokens=512, temperature=.7, top_p=1., seed=1):
        raw = prompt.encode('utf-8')
        with self._lock:
            self._require_open()
            self._check(self.lib.kyse_begin(self._handle, raw, len(raw), max_tokens, temperature, top_p, seed))

    def next_token(self):
        """Return owned token bytes, or None at EOS/limit. May return b'' for an empty token."""
        with self._lock:
            self._require_open()
            status = self._check(self.lib.kyse_next(self._handle, self._token))
            if status == 0:
                return None
            return bytes(self.ffi.buffer(self._token.data, self._token.size))

    def stop(self):
        with self._lock:
            self._require_open()
            self._check(self.lib.kyse_stop(self._handle))

    def usage(self):
        with self._lock:
            self._require_open()
            self._check(self.lib.kyse_usage(self._handle, self._usage))
            return {'prompt_tokens': self._usage.prompt_tokens,
                    'completion_tokens': self._usage.completion_tokens,
                    'total_tokens': self._usage.prompt_tokens + self._usage.completion_tokens,
                    'finish_reason': 'length' if self._usage.length else 'stop'}

    def close(self):
        with self._lock:
            if not self.closed:
                handle, self._handle = self._handle, None
                self._finalizer.detach()
                self._check(self.lib.kyse_close(handle))

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()
