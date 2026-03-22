# Jam Dragon

## How to build
clang/gcc, open mp, SDL2 stuff

## Usage

Run the simulation with:
```bash
./draw2 [max_frames] [--fluid x y w h] [--ghost x y w h]
```

- `max_frames`: (Optional) The maximum number of frames to simulate before exiting. Default: -1 (infinite).
- `--fluid x y w h` or `-f x y w h`: Spawn a fluid box at (x, y) with width w and height h. Multiple fluid boxes can be specified.
- `--ghost x y w h` or `-g x y w h`: Spawn a static ghost box at (x, y) with width w and height h.

If no boxes are provided, the simulation will start with no particles.
