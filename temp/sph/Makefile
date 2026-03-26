CC = g++
OBJ_NAME = draw2

# 1. Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

# 2. Find all .cpp files in src/
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# 3. Map src/*.cpp to obj/*.o
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# 4. Flags
# -I$(INC_DIR) tells the compiler to look in the include folder for .hpp files
CFLAGS = -std=c++20 -Ofast -march=skylake-avx512 -ftree-vectorize -fvect-cost-model=unlimited -ffast-math -funroll-loops -fopt-info-vec-optimized -flto -fopenmp -Wall -I$(INC_DIR)

# SDL Check
HAS_SDL := $(shell pkg-config --exists sdl2 SDL2_image && echo yes || echo no)

ifeq ($(HAS_SDL),yes)
	SDL_CFLAGS = $(shell pkg-config --cflags sdl2 SDL2_image) -DUSE_SDL
	SDL_LDFLAGS = $(shell pkg-config --libs sdl2 SDL2_image)
else
	SDL_CFLAGS =
	SDL_LDFLAGS =
endif

CFLAGS += $(SDL_CFLAGS)
LDFLAGS = $(SDL_LDFLAGS) -lpthread -fopenmp -flto

# Default Target
all: $(OBJ_DIR) $(OBJ_NAME)

# Create the obj directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Final Linker Step + Auto-Launch
$(OBJ_NAME) : $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(OBJ_NAME)
	@echo "----------------------------"
	@echo "Build Successful. Launching..."
	@echo "----------------------------"
	# ./$(OBJ_NAME)

# Pattern Rule: Compile src/%.cpp into obj/%.o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(OBJ_NAME)

.PHONY: all clean
