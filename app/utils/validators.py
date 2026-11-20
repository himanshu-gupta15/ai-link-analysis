"""URL validation and SSRF protection.

Every URL the service fetches (the submitted URL, every redirect hop, extra
internal pages and competitor websites) goes through ``ensure_public_url``:

1. Only ``http``/``https``, no embedded credentials, only allowed ports.
2. Hostname must be a real public domain (no ``localhost``, ``.local``, ``.internal`` ...).
3. Every IP address the hostname resolves to must be globally routable – private,
   loopback, link-local (cloud metadata 169.254.169.254), CGNAT, multicast and
   reserved ranges are rejected, including IPv4-mapped IPv6 forms.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit, urlunsplit

BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".intranet", ".lan", ".home", ".corp", ".localdomain")
BLOCKED_HOSTS = {"localhost", "localhost.localdomain", "metadata.google.internal"}


class InvalidURLError(ValueError):
    """The URL is malformed or uses an unsupported scheme/port."""


class BlockedURLError(ValueError):
    """The URL points to a private, internal or otherwise disallowed address."""


class DNSResolutionError(ValueError):
    """The hostname could not be resolved."""


def normalize_url(raw: str) -> str:
    """Trim, add ``https://`` when the scheme is missing and validate the shape."""
    url = (raw or "").strip()
    if not url:
        raise InvalidURLError("URL must not be empty")
    if len(url) > 2048:
        raise InvalidURLError("URL is too long (max 2048 characters)")
    if "://" not in url:
        url = "https://" + url

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise InvalidURLError(f"Unsupported URL scheme '{parts.scheme}'. Only http and https are allowed")
    if parts.username or parts.password:
        raise InvalidURLError("URLs containing credentials are not allowed")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise InvalidURLError("URL must contain a hostname")
    try:
        port = parts.port
    except ValueError:
        raise InvalidURLError("URL contains an invalid port") from None

    # Reject obviously invalid hostnames early (spaces, underscores, empty labels).
    if not _is_ip_literal(host):
        try:
            host.encode("idna")
        except UnicodeError:
            raise InvalidURLError(f"Invalid hostname '{host}'") from None
        labels = host.split(".")
        if len(labels) < 2 or any(not lbl or len(lbl) > 63 for lbl in labels):
            raise InvalidURLError(f"'{host}' is not a valid public domain name")
        if any(ch.isspace() or ch in "_<>\"'{}|\\^`" for ch in host):
            raise InvalidURLError(f"Invalid hostname '{host}'")

    netloc = host if port is None else f"{host}:{port}"
    if ":" in host and not host.startswith("["):  # IPv6 literal
        netloc = f"[{host}]" if port is None else f"[{host}]:{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def is_public_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


async def resolve_host(host: str, port: int) -> list[str]:
    """Resolve ``host`` to all of its IP addresses (async, non-blocking)."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise DNSResolutionError(f"Could not resolve hostname '{host}'") from exc
    return sorted({info[4][0] for info in infos})


async def ensure_public_url(url: str, allowed_ports: set[int]) -> str:
    """Validate ``url`` and make sure it resolves only to public IP addresses.

    Returns the normalized URL. Raises ``InvalidURLError``, ``BlockedURLError``
    or ``DNSResolutionError``.
    """
    url = normalize_url(url)
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    port = parts.port or (443 if parts.scheme == "https" else 80)

    if port not in allowed_ports:
        raise BlockedURLError(f"Port {port} is not allowed")
    if host in BLOCKED_HOSTS or host.endswith(BLOCKED_HOST_SUFFIXES):
        raise BlockedURLError(f"Access to internal host '{host}' is not allowed")

    if _is_ip_literal(host):
        addresses = [host.strip("[]")]
    else:
        addresses = await resolve_host(host, port)
    if not addresses:
        raise DNSResolutionError(f"Could not resolve hostname '{host}'")
    blocked = [a for a in addresses if not is_public_ip(a)]
    if blocked:
        raise BlockedURLError(
            f"Access to private or internal network addresses is not allowed ({host} -> {', '.join(blocked)})"
        )
    return url
