import sys

from omnidrive_v5.engine.pipeline import main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--stage", "train", *sys.argv[1:]]
    main()
