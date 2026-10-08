# SPDX-License-Identifier: Apache-2.0
"""Stream OpenAI requests to the least-busy of two independent TP1 engines."""

import argparse
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate",
               "proxy-authorization", "te", "trailer", "transfer-encoding",
               "upgrade", "host", "content-length"}


def create_app(upstreams: list[str]) -> FastAPI:
    if len(upstreams) != 2 or len(set(upstreams)) != 2:
        raise ValueError("Two distinct TP1 upstreams are required")
    active = [0, 0]
    next_worker = 0

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=10)) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        for url in upstreams:
            try:
                response = await app.state.client.get(url + "/health")
                if response.status_code != 200:
                    return JSONResponse({"ready": False}, status_code=503)
            except httpx.HTTPError:
                return JSONResponse({"ready": False}, status_code=503)
        return {"ready": True, "active": active}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE"])
    async def proxy(path: str, request: Request):
        nonlocal next_worker
        # No await between selection and reservation: this is atomic in the
        # single event loop. Streaming requests stay reserved until closed.
        worker = min(range(2), key=lambda i: (active[i], i != next_worker))
        next_worker = 1 - worker
        active[worker] += 1
        response = None
        try:
            headers = {k: v for k, v in request.headers.items()
                       if k.lower() not in HOP_HEADERS}
            outgoing = app.state.client.build_request(
                request.method, upstreams[worker].rstrip("/") + "/v1/" + path,
                params=request.query_params, headers=headers,
                content=await request.body(),
            )
            response = await app.state.client.send(outgoing, stream=True)
        except BaseException as error:
            active[worker] -= 1
            if response is not None:
                await response.aclose()
            if isinstance(error, httpx.HTTPError):
                return JSONResponse({"error": "Upstream unavailable"}, status_code=502)
            raise

        async def stream():
            try:
                async for data in response.aiter_raw():
                    yield data
            finally:
                await response.aclose()
                active[worker] -= 1

        return StreamingResponse(
            stream(), status_code=response.status_code,
            headers={k: v for k, v in response.headers.items()
                     if k.lower() not in HOP_HEADERS},
        )

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", action="append", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(create_app(args.upstream), host=args.host, port=args.port)
