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

all: qwen bench

qwen: main.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ main.c $(SRC) $(LDFLAGS) -lm

bench: bench.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ bench.c $(SRC) $(LDFLAGS) -lm

# make export HF=path/to/hf/checkpoint
HF ?= ../models/qwen3.5-0.8b
export:
	python3 export.py $(HF) .

test:
	python3 tests/test_model.py

clean:
	rm -f qwen bench

.PHONY: all export test clean
