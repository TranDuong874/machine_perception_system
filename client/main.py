from __future__ import annotations

from config import load_client_config
from runtime import ClientRuntime


def main() -> None:
    config = load_client_config()
    ClientRuntime(config).run()


if __name__ == "__main__":
    main()
