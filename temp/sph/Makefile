ACPP ?= acpp
TEST_CXX ?= g++
ACPP_TARGETS ?= omp
SMOKE_FRAMES ?= 2
SMOKE_RENDER ?= density

SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
DENSITY_ONLY_OBJ_DIR := $(BUILD_DIR)/density-only
DENSITY_VELOCITY_OBJ_DIR := $(BUILD_DIR)/density-velocity
TEST_OBJ_DIR := $(BUILD_DIR)/tests

COMMON_SOURCES := \
	$(SRC_DIR)/main.cpp \
	$(SRC_DIR)/cli.cpp \
	$(SRC_DIR)/metadata.cpp \
	$(SRC_DIR)/sycl.cpp \
	$(SRC_DIR)/floaters.cpp \
	$(SRC_DIR)/graphics.cpp \
	$(SRC_DIR)/spatial.cpp \
	$(SRC_DIR)/math.cpp \
	$(SRC_DIR)/gravity.cpp \
	$(SRC_DIR)/render.cpp \
	$(SRC_DIR)/logging.cpp

DENSITY_ONLY_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(DENSITY_ONLY_OBJ_DIR)/%.o,$(COMMON_SOURCES))
DENSITY_VELOCITY_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(DENSITY_VELOCITY_OBJ_DIR)/%.o,$(COMMON_SOURCES))
TEST_SOURCES := tests/core_tests.cpp $(SRC_DIR)/cli.cpp $(SRC_DIR)/metadata.cpp $(SRC_DIR)/math.cpp $(SRC_DIR)/render.cpp
TEST_HEADERS := $(INC_DIR)/cli.hpp $(INC_DIR)/math.hpp $(INC_DIR)/metadata.hpp $(INC_DIR)/render.hpp $(INC_DIR)/settings.hpp $(INC_DIR)/struct.hpp

SDL_CFLAGS := $(shell pkg-config --cflags sdl2 2>/dev/null || sdl2-config --cflags 2>/dev/null)
SDL_LDFLAGS := $(shell pkg-config --libs sdl2 2>/dev/null || sdl2-config --libs 2>/dev/null)

CPPFLAGS += -I$(INC_DIR) $(SDL_CFLAGS)
CXXFLAGS += -std=c++20 -O3 -g -ffast-math -fno-finite-math-only -fopenmp -Wall -Wextra -Wpedantic
LDFLAGS += $(SDL_LDFLAGS) -fopenmp -pthread
ACPP_FLAGS := --acpp-targets="$(ACPP_TARGETS)"
DENSITY_ONLY_FLAGS := -DSPH_DENSITY_ONLY
DENSITY_VELOCITY_FLAGS := -DSPH_DENSITY_VELOCITY

ifeq ($(strip $(SDL_LDFLAGS)),)
else
CPPFLAGS += -DUSE_SDL
endif

.PHONY: all variants clean check smoke smoke-density-only smoke-density-velocity draw2 draw2-density-only draw2-density-velocity

all: draw2

variants: draw2 draw2-density-velocity

draw2: draw2-density-only
	ln -sfn draw2-density-only $@

draw2-density-only: $(DENSITY_ONLY_OBJECTS)
	$(ACPP) $(ACPP_FLAGS) $^ $(LDFLAGS) -o $@

draw2-density-velocity: $(DENSITY_VELOCITY_OBJECTS)
	$(ACPP) $(ACPP_FLAGS) $^ $(LDFLAGS) -o $@

$(DENSITY_ONLY_OBJ_DIR):
	mkdir -p $@

$(DENSITY_VELOCITY_OBJ_DIR):
	mkdir -p $@

$(DENSITY_ONLY_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(DENSITY_ONLY_OBJ_DIR)
	$(ACPP) $(ACPP_FLAGS) $(CPPFLAGS) $(CXXFLAGS) $(DENSITY_ONLY_FLAGS) -MMD -MP -c $< -o $@

$(DENSITY_VELOCITY_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(DENSITY_VELOCITY_OBJ_DIR)
	$(ACPP) $(ACPP_FLAGS) $(CPPFLAGS) $(CXXFLAGS) $(DENSITY_VELOCITY_FLAGS) -MMD -MP -c $< -o $@

$(TEST_OBJ_DIR)/core_tests: $(TEST_SOURCES) $(TEST_HEADERS)
	@mkdir -p $(@D)
	$(TEST_CXX) -std=c++20 -O2 -Wall -Wextra -Wpedantic -I$(INC_DIR) $(TEST_SOURCES) -o $@

check: $(TEST_OBJ_DIR)/core_tests
	$(TEST_OBJ_DIR)/core_tests

smoke: smoke-density-only smoke-density-velocity

smoke-density-only: draw2-density-only
	@set -eu; \
	tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT; \
	SPH_DATA_ROOT="$$tmp" ./draw2-density-only --headless --render "$(SMOKE_RENDER)" --frames "$(SMOKE_FRAMES)" $${SMOKE_ARGS:---fluid 120 120 24 24}; \
	data=$$(find "$$tmp" -name sim_data.bin -type f -print -quit); \
	test -n "$$data"; \
	python3 -c 'import json, pathlib, sys; import numpy as np; value=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert value["model_variant"] == "density-only"; frame=np.fromfile(sys.argv[2], dtype=np.float32).reshape((-1, 4, 400, 400)); assert frame.shape[0] == int(sys.argv[3]); assert np.any(frame[:, 1:3] != 0)' "$$(dirname "$$data")/metadata.json" "$$data" "$(SMOKE_FRAMES)"

smoke-density-velocity: draw2-density-velocity
	@set -eu; \
	tmp=$$(mktemp -d); \
	trap 'rm -rf "$$tmp"' EXIT; \
	SPH_DATA_ROOT="$$tmp" ./draw2-density-velocity --headless --render "$(SMOKE_RENDER)" --frames "$(SMOKE_FRAMES)" $${SMOKE_ARGS:---fluid 120 120 24 24}; \
	data=$$(find "$$tmp" -name sim_data.bin -type f -print -quit); \
	test -n "$$data"; \
	python3 -c 'import json, pathlib, sys; import numpy as np; value=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert value["model_variant"] == "density-velocity"; frame=np.fromfile(sys.argv[2], dtype=np.float32).reshape((-1, 4, 400, 400)); assert frame.shape[0] == int(sys.argv[3]); assert np.any(frame[:, 1:3] != 0)' "$$(dirname "$$data")/metadata.json" "$$data" "$(SMOKE_FRAMES)"

clean:
	rm -rf $(BUILD_DIR) obj draw2 draw2-density-only draw2-density-velocity

-include $(DENSITY_ONLY_OBJECTS:.o=.d)
-include $(DENSITY_VELOCITY_OBJECTS:.o=.d)
