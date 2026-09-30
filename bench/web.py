"""HTTP for the bench scripts: a read that tries again when the network or a server fails for a moment."""

import sys
import time
import urllib.error
import urllib.request


def read(req: urllib.request.Request | str, timeout: float = 600, tries: int = 4) -> bytes:
    """The body of `req`.

    A dropped connection, a timeout, 429 or a 5xx is tried again after 2, 4 and 8 s. Any other
    HTTP error, such as a 422 for a bad request, is raised at once.
    """
    for n in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if (exc.code != 429 and exc.code < 500) or n == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if n == tries - 1:
                raise
        # A retried request's time includes the wait, so the caller's latency for it is too long.
        print(f"retrying {getattr(req, 'full_url', req)} in {2 ** (n + 1)} s", file=sys.stderr, flush=True)
        time.sleep(2 ** (n + 1))
    raise AssertionError("unreachable")
