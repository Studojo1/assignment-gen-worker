Assignment Gen Worker
=====================

Go worker service that consumes assignment generation jobs from RabbitMQ.

Overview
--------

This worker service processes assignment generation jobs by:

- Consuming jobs from RabbitMQ (``assignment-gen.jobs`` queue)
- Handling ``assignment-gen``, ``outline-gen``, and ``outline-edit`` job types
- Invoking Python assignment generator
- Uploading generated documents to Azure Blob Storage (for assignment-gen)
- Publishing result events to RabbitMQ

Technology
----------

- Go 1.23

Environment Variables
---------------------

- ``RABBITMQ_URL`` - RabbitMQ connection URL
- ``RESULTS_EXCHANGE`` - Results exchange name (default: ``cp.results``)
- ``USE_LOCALSTACK`` - Use LocalStack (true/false)
- ``LOCALSTACK_ENDPOINT`` - LocalStack endpoint URL
- ``AZURE_STORAGE_ACCOUNT_NAME`` - Blob storage account name
- ``AZURE_STORAGE_ACCOUNT_KEY`` - Blob storage account key
- ``AZURE_STORAGE_CONTAINER_NAME`` - Container name (default: ``assignments``)
- ``PYTHON_PATH`` - Python executable path
- ``ASSIGNMENT_GEN_SCRIPT_PATH`` - Path to Python assignment generator script
- ``AZURE_OPENAI_ENDPOINT`` - Azure OpenAI endpoint (all OpenAI model calls go through Azure)
- ``AZURE_OPENAI_API_KEY`` - Azure OpenAI API key
- ``AZURE_OPENAI_MINI_DEPLOYMENT`` - Chat deployment (default: ``gpt-4o-mini``)
- ``AZURE_OPENAI_EMBEDDING_DEPLOYMENT`` - Embeddings deployment (default: ``text-embedding-3-small``)
- ``OPENAI_API_VERSION`` - Azure OpenAI API version (default: ``2024-10-21``)
- ``ANTHROPIC_API_KEY`` - Anthropic API key
- ``REPHRASY_API_KEY`` - Rephrasy API key
