CC     ?= clang
CFLAGS ?= -O3 -march=native -Wall -Wextra -std=gnu23
SRC     = model.c tokenizer.c sampler.c ops_ref.c backend_cpu.c
HDR     = model.h backend.h tokenizer.h sampler.h util.h

ifeq ($(shell uname),Darwin)
  SRC     += backend_accelerate.c backend_metal.m
  HDR     += kernels.metal
  CFLAGS  += -fobjc-arc
  LDFLAGS += -framework Metal -framework Foundation -framework Accelerate
endif

SRC := $(addprefix src/,$(SRC))
HDR := $(addprefix src/,$(HDR))

ifeq ($(shell uname),Darwin)
  LIBRARY = build/libkyse.dylib
  SHARED_FLAGS = -dynamiclib
  LIBRARY_SRC = src/library_pool.m
else
  LIBRARY = build/libkyse.so
  SHARED_FLAGS = -shared
endif

all: qwen bench qwen-worker library

qwen: src/main.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ src/main.c $(SRC) $(LDFLAGS) -lm

bench: src/bench.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ src/bench.c $(SRC) $(LDFLAGS) -lm

qwen-worker: src/worker.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ src/worker.c $(SRC) $(LDFLAGS) -lm

library: $(LIBRARY)

$(LIBRARY): src/kyse.c src/kyse.h $(SRC) $(HDR) $(LIBRARY_SRC)
	mkdir -p build
	$(CC) $(CFLAGS) -DKYSE_LIBRARY -fPIC -fvisibility=hidden $(SHARED_FLAGS) -o $@ src/kyse.c $(SRC) $(LIBRARY_SRC) $(LDFLAGS) -pthread -lm

# make export HF=path/to/hf/checkpoint
HF ?= ../models/qwen3.5-0.8b
export:
	python3 scripts/export.py $(HF) .

test:
	python3 tests/test_model.py

SERVER_PYTHON ?= python3
PIARA_DIR ?= ../piara
test-server: qwen-worker library
	PYTHONPATH="$(abspath $(PIARA_DIR)/..)" KYSE_ROOT="$(CURDIR)" $(SERVER_PYTHON) -m unittest discover -s "$(PIARA_DIR)/tests" -v

clean:
	rm -f qwen bench qwen-worker $(LIBRARY)

.PHONY: all library export test test-server clean
