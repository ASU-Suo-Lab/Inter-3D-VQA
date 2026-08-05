from __future__ import annotations

import json

from geovlm_intersection.models.architecture import shape_smoke_forward


def main() -> None:
    print(json.dumps(shape_smoke_forward(), indent=2))


if __name__ == "__main__":
    main()
