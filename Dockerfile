FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy workspace config and source
COPY pyproject.toml uv.lock .python-version ./
COPY src/ src/

# Install production deps only (no dev group)
RUN uv sync --frozen --no-group dev

# Override CMD per use case (scripts/, module entrypoints, etc.)
CMD ["uv", "run", "python", "-c", "print('arctic-showcase ready')"]
