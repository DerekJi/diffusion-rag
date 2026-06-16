# Copilot Instructions for VedaAgent

## Repository Overview

This is **VedaAgent**, a Python-based agent framework designed for autonomous operation and interaction. It provides a core kernel, memory management, tool integration, and a TUI (Terminal User Interface) for interaction.

**Key Technologies:**
- Backend: Python 3.11+, FastAPI, SQLAlchemy (or similar ORM), SQLite (or other database), Poetry for dependency management.
- Frontend: TypeScript/JavaScript (for `tui/`), likely using Vitest for testing.
- Testing: Pytest (backend), Vitest (frontend), Playwright (E2E if applicable).
- Architecture: Modular Python services, Agentic architecture with tools and memory.
- Cloud: Docker (via `docker-compose.yml`).

## Critical Prerequisites

**IMPORTANT:** This repository uses Poetry for Python dependency management and potentially `npm`/`pnpm`/`yarn` for frontend dependencies.

- **Poetry:** Ensure Poetry is installed and configured.
- **Node.js/npm:** If working with the `tui/` frontend, ensure Node.js and a package manager (npm, pnpm, or yarn) are installed.

## Build & Test Commands

### Backend (Python API)

**ALWAYS run commands in this exact order:**

1.  **Install dependencies**:
    ```bash
    poetry install
    # Takes: ~1-2 minutes
    ```

2.  **Run backend unit tests**:
    ```bash
    poetry run pytest tests/
    # Takes: ~1 minute
    ```

### Frontend (TUI/Web UI)

**ALWAYS run from `tui/` directory if applicable:**

1.  **Install dependencies**:
    ```bash
    cd tui/
    npm install # Or pnpm install, yarn install
    # Takes: ~1-2 minutes
    ```

2.  **Lint check**:
    ```bash
    cd tui/
    npm run lint # Or pnpm lint, yarn lint
    # Fix automatically:
    npm run lint:fix # Or pnpm lint:fix, yarn lint:fix
    ```

3.  **Run unit tests**:
    ```bash
    cd tui/
    npm run test # Or pnpm test, yarn test
    # Watch mode:
    npm run test:watch # Or pnpm test:watch, yarn test:watch
    # Coverage:
    npm run test:coverage # Or pnpm test:coverage, yarn test:coverage
    ```

### E2E Tests (Playwright)

**Location:** `tests/integration/e2e/` (if E2E tests exist)

**Before first run (example):**
```bash
poetry run playwright install
```

**Run E2E tests (example):**
```bash
poetry run pytest tests/integration/e2e/
```

## Local Development

### Backend API

**Option 1 - Using Poetry (from repo root):**
```bash
poetry run uvicorn kernel.web.app:app --reload --port 8000
```

**Option 2 - Using Docker Compose (from repo root):**
```bash
docker-compose up veda-web
# To include sidecar:
docker-compose --profile sidecars up veda-web veda-sidecar
```

**Runs on:** `http://localhost:8000`

**Environment Variables:** Set these in a `.env` file or pass them on the command line. (Do NOT commit real API keys to version control). Refer to `docker-compose.yml` for common variables like `OPENAI_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`, `MEMORY_ENABLED`, `SIDECARS_ENABLED`, `SHELL_SANDBOX`, `VEDA_API_KEYS`, `CORS_ALLOW_ORIGINS`.

### Frontend UI (TUI/Web)

```bash
cd tui/
npm run dev # Or pnpm dev, yarn dev
```

**Runs on:** `http://localhost:5173` (default for Vite development server)

## Project Structure

### Key Directories

```
/
├── .github/                # GitHub AI configs (copilot-instructions, prompts, naming conventions)
├── config/                 # Application configuration files (e.g., `sidecars.yaml`)
├── data/                   # Persistent data storage
├── docs/                   # Documentation and reports
├── harness/                # Test harness for scenarios
├── kernel/                 # Core agent logic, LLM interaction, graph, DB
├── logs/                   # Application logs
├── memory/                 # Memory management (short-term, long-term, retrieval)
├── presentation/           # Web presentation layer
├── session/                # Session management
├── sidecar/                # Sidecar worker logic
├── tests/                  # All tests (unit, integration, E2E)
├── tools/                  # Tool definitions and adapters
├── tui/                    # Terminal User Interface (Frontend)
├── Dockerfile              # Docker build instructions
├── docker-compose.yml      # Docker Compose setup for local development/deployment
├── pyproject.toml          # Poetry project configuration and dependencies
├── README.md               # Project overview
└── ...
```

### Configuration Files

-   **Root:** `pyproject.toml`, `poetry.lock`, `README.md`, `.env` (local environment variables)
-   **Config Directory:** `config/sidecars.yaml`
-   **Frontend:** `tui/package.json`, `tui/tsconfig.json`, `tui/vitest.config.ts`, `tui/eslint.config.mjs` (if exists)
-   **Linting/Formatting:** `.editorconfig`, `.prettierrc` (if exists)

## CI/CD & Validation

### Pull Request Checks

**Workflow:** `.github/workflows/main.yml` (or similar) - will need to be configured for Python.

**What runs (example):**
1.  Poetry install and dependency caching
2.  Python linting (`ruff`, `flake8`)
3.  Backend unit tests (`pytest`)
4.  Frontend linting (`eslint`, `prettier`) - if `tui/` exists
5.  Frontend unit tests (`vitest`) - if `tui/` exists
6.  E2E tests (Playwright) - if configured
7.  Security scanning (e.g., `bandit`, `snyk`)

### Common Build/Test Failures & Solutions

-   **Dependency errors:** Ensure `poetry install` or `npm install` (for `tui/`) runs successfully. Check `pyproject.toml` or `package.json`.
-   **Linting/formatting errors:** Run `poetry run ruff check --fix` (Python) or `npm run lint:fix` / `npm run prettier:fix` (Frontend).
-   **Test failures:** Run `poetry run pytest` or `npm run test` locally to debug.

## Architecture Patterns

**VedaAgent** employs a modular, agentic architecture:
-   **Kernel:** The core of the agent, handling LLM interactions, graph state, and database operations.
-   **Memory:** Manages different tiers of memory (short-term, long-term) for context and retrieval.
-   **Tools:** Provides an extensible framework for integrating external capabilities (file system, shell, code analysis, custom tools).
-   **Session:** Manages conversation state and context.
-   **Sidecar:** Optional worker processes for asynchronous tasks.
-   **UI:** Terminal User Interface (`tui/`).

## Testing Best Practices

### Backend Unit Tests (Pytest)

**Framework & Libraries:**
-   **Pytest** - Testing framework
-   **unittest.mock** - Mocking library (built-in Python)
-   **assert** statements - Standard Python assertions

**When writing unit tests:**
1.  Use `test_` prefix for test files and functions.
2.  Use `pytest.fixture` for test setup/teardown.
3.  Use `unittest.mock.patch` for mocking dependencies.
4.  Use standard `assert` statements for assertions.

**Example test pattern:**
```python
import pytest
from unittest.mock import MagicMock

def test_function_name_when_condition_should_expected_behavior():
    # Arrange
    mock_dependency = MagicMock()
    sut = SystemUnderTest(mock_dependency)

    # Act
    result = sut.method()

    # Assert
    assert result == expected_value
    mock_dependency.some_call.assert_called_once()
```

### Frontend Unit Tests (Vitest)

-   Use Vitest for `tui/` component and service testing.
-   Follow React testing best practices (React Testing Library, Vitest).
-   Run `npm run test` (from `tui/` directory) for unit tests.

## Security & Best Practices

1.  **Never commit secrets** - Use `.env` locally, environment variables in deployed environments.
2.  **Run security audit** - Integrate tools like Bandit for Python code scanning.
3.  **Follow PEP 8** - Python code style is enforced via linters (e.g., Ruff).
4.  **Validate configuration** - Ensure configuration is validated on startup.
5.  **Health checks** - Implement health checks for services (e.g., `/health` endpoint).

## Useful Health Check Endpoints

-   `/health` - Overall health of the VedaAgent API (200 if healthy)
-   `/version` - Application version and build information

## When Making Changes

1.  **For Python code:**
    -   Follow PEP 8 and project-specific naming conventions (`.github/python-naming-conventions.instructions.md`).
    -   Run linters (`poetry run ruff check --fix`).
    -   Ensure unit tests pass (`poetry run pytest`).

2.  **For TypeScript/JavaScript (tui/) code:**
    -   Run `npm run lint:fix` and `npm run prettier:fix`.
    -   Follow React framework style guide.
    -   Update tests when changing components.
    -   Build locally: `cd tui/ && npm run build`.

3.  **For Docker/Docker Compose:**
    -   Ensure `docker-compose.yml` is up-to-date with services and environment variables.
    -   Build and test Docker images locally.

4.  **Always validate before pushing:**
    ```bash
    # Backend
    poetry install && poetry run ruff check && poetry run pytest

    # Frontend (if applicable)
    cd tui/ && npm install && npm run lint && npm run test && npm run build
    ```

## Trust These Instructions

These instructions are comprehensive and validated. Only perform additional searches if:
-   The information here is incomplete for your specific task
-   You encounter errors not mentioned in this document
-   You need details about a specific integration or feature not covered
-   The repository has been significantly restructured since these instructions were written

For most common tasks (building, testing, adding features, fixing bugs), this document contains all necessary information.
