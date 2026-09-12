# Dependencies first, in a layer of their own: the source changes on every build and
# the wheel downloads do not.
FROM python:3.13-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml ./
# An empty package so `pip install .` can resolve the project before the real sources
# arrive. Without it pip has nothing to read the dependency list from.
RUN mkdir -p app && touch app/__init__.py && \
    pip install --prefix=/install .

COPY app ./app
RUN pip install --prefix=/install --no-deps .

FROM python:3.13-slim

# Not root. This process talks to a broker and a cipher, and holds a token for each;
# there is no reason for it to be able to write anything outside its own directory.
RUN useradd --create-home --uid 10001 mcp

COPY --from=build /install /usr/local
WORKDIR /app
COPY app ./app

USER mcp
EXPOSE 8000

# Read at start up, not baked in: the host and port belong to wherever this runs.
ENV HOST=0.0.0.0 \
    PORT=8000

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT}"]
