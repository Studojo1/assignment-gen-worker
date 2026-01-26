FROM golang:1.23-alpine AS builder
WORKDIR /app
COPY go.mod go.sum* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags='-w -s' -o worker ./cmd/worker

# Use Debian-based Python image for better package compatibility
FROM python:3.13-slim
WORKDIR /app

# Install system dependencies needed for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies needed by assignment-gen
# These match the dependencies in services/assignment-gen/pyproject.toml
RUN pip install --no-cache-dir \
    "anthropic>=0.73.0" \
    "beautifulsoup4>=4.14.2" \
    "ddgs>=9.9.1" \
    "langchain>=1.0.7" \
    "langchain-classic>=0.1.0" \
    "langchain-community>=0.4.1" \
    "langchain-openai>=1.0.3" \
    "langgraph>=1.0.3" \
    "matplotlib>=3.8.0" \
    "math2docx>=0.1.0" \
    "numpy>=1.26.0" \
    "openai>=2.8.0" \
    "pydantic>=2.12.4" \
    "python-docx>=1.2.0" \
    "python-dotenv>=1.2.1" \
    "PyPDF2>=3.0.0" \
    "pdfplumber>=0.11.0" \
    "Pillow>=10.0.0" \
    "requests>=2.32.0" \
    "python-dateutil>=2.8.0" \
    "openpyxl>=3.1.0" \
    "scipy>=1.11.0" \
    "scikit-learn>=1.3.0" \
    "pandas>=2.0.0" \
    "reportlab>=4.0.0" \
    "nest-asyncio>=1.5.8"

# Copy the Go binary
COPY --from=builder /app/worker .
RUN chown -R appuser:appuser /app
USER appuser

CMD ["./worker"]
