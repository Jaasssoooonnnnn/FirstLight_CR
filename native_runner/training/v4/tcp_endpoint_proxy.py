#!/usr/bin/env python3
"""Expose loopback-only native-engine endpoints on a cluster interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import socket
from collections.abc import Sequence


_COPY_CHUNK_BYTES = 1 << 20


async def _copy_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(_COPY_CHUNK_BYTES):
        writer.write(data)
        await writer.drain()
    if writer.can_write_eof():
        writer.write_eof()
        await writer.drain()


async def _handle_connection(
    client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter, *, target_host: str, target_port: int
) -> None:
    upstream_writer: asyncio.StreamWriter | None = None
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(target_host, target_port)
        client_socket = client_writer.get_extra_info("socket")
        upstream_socket = upstream_writer.get_extra_info("socket")
        for stream_socket in (client_socket, upstream_socket):
            if stream_socket is not None:
                stream_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        await asyncio.gather(_copy_stream(client_reader, upstream_writer), _copy_stream(upstream_reader, client_writer))
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        client_writer.close()
        if upstream_writer is not None:
            upstream_writer.close()
        await asyncio.gather(
            client_writer.wait_closed(),
            *((upstream_writer.wait_closed(),) if upstream_writer is not None else ()),
            return_exceptions=True,
        )


async def run_proxies(*, bind_host: str, listen_base: int, target_host: str, target_base: int, count: int) -> None:
    servers: list[asyncio.Server] = []
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    try:
        for offset in range(count):
            target_port = target_base + offset

            async def handler(
                reader: asyncio.StreamReader, writer: asyncio.StreamWriter, *, port: int = target_port
            ) -> None:
                await _handle_connection(reader, writer, target_host=target_host, target_port=port)

            server = await asyncio.start_server(handler, bind_host, listen_base + offset, backlog=512, limit=4 << 20)
            servers.append(server)
        print(
            json.dumps(
                {
                    "bind_host": bind_host,
                    "count": count,
                    "listen_base": listen_base,
                    "ready": True,
                    "target_base": target_base,
                    "target_host": target_host,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        await stop.wait()
    finally:
        for server in servers:
            server.close()
        await asyncio.gather(*(server.wait_closed() for server in servers), return_exceptions=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", required=True)
    parser.add_argument("--listen-base", required=True, type=int)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-base", required=True, type=int)
    parser.add_argument("--count", required=True, type=int)
    args = parser.parse_args(argv)
    if args.count <= 0:
        parser.error("--count must be positive")
    for name in ("listen_base", "target_base"):
        base = getattr(args, name)
        if base <= 0 or base + args.count > 65536:
            parser.error(f"--{name.replace('_', '-')} range exceeds TCP ports")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    asyncio.run(
        run_proxies(
            bind_host=args.bind_host,
            listen_base=args.listen_base,
            target_host=args.target_host,
            target_base=args.target_base,
            count=args.count,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
