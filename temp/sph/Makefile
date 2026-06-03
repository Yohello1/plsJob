CC = acpp
OBJ_NAME = draw2

# 1. Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

# 2. Find all .cpp files in src/
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# 3. Map src/*.cpp to obj/*.o
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# 4. SDL Check
HAS_SDL := $(shell pkg-config --exists sdl2 SDL2_image && echo yes || echo no)

# 5. Base Flags (Common to both Compilation and Linking)
# -flto and -fuse-ld=lld MUST be visible to both steps!
COMMON_FLAGS = -std=c++20 -O3 -g -ffast-math -funroll-loops -flto -fuse-ld=lld -fopenmp -march=native

# Compiler-only flags
CFLAGS = $(COMMON_FLAGS) -Wall -I$(INC_DIR)

# Linker-only flags
LDFLAGS = $(COMMON_FLAGS) -lpthread -Wl,-rpath,/u2/s23adhik/llvm-custom/lib/x86_64-unknown-linux-gnu

# 6. Conditionally Add SDL2 if found
ifeq ($(HAS_SDL),yes)
    CFLAGS += $(shell pkg-config --cflags sdl2 SDL2_image) -DUSE_SDL
    LDFLAGS += $(shell pkg-config --libs sdl2 SDL2_image)
endif

# Default Target
all: $(OBJ_DIR) $(OBJ_NAME)

# Create the obj directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Final Linker Step
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
