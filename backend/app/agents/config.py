from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    # Master switch. When false, the router is not mounted, MCP servers are
    # not contacted, and no background tasks run.
    enabled: bool = False

    # Built-in MCP server URL. The mcp-server container speaks JSON-RPC 2.0.
    builtin_mcp_url: str = "http://mcp-server:8765/mcp"

    # Comma-separated extra MCP servers users can plug in (URL[|name]).
    # Example: "http://my-tools:9000/mcp|my-tools,http://other:9001/mcp"
    extra_mcp_servers: str = ""

    # Shared secret used to mint short-lived JWTs for MCP calls. Distinct
    # from the main app secret so revocation is independent.
    mcp_jwt_secret: str = "change-me-in-production"
    mcp_jwt_ttl_seconds: int = 600

    # Which provider validates MCP bearer tokens. "internal" accepts the
    # HS256 tokens minted by the in-process agent runtime (keeps the
    # built-in AI Agents working). "workos" validates tokens issued by
    # WorkOS AuthKit via JWKS — required for external clients like Claude
    # to authenticate over OAuth 2.1.
    mcp_auth_provider: str = "internal"

    # WorkOS AuthKit settings. Only consulted when mcp_auth_provider is
    # "workos". The issuer is your AuthKit domain (e.g.
    # https://<subdomain>.authkit.app). The JWKS URL is published at
    # `${issuer}/oauth2/jwks` for AuthKit-issued tokens.
    workos_issuer: str = ""
    workos_jwks_url: str = ""
    workos_audience: str = ""

    # Public URL where the MCP server is reachable from outside the
    # Docker network. Returned as the `resource` field of the OAuth
    # protected-resource discovery document so MCP clients send tokens
    # bound to this resource.
    mcp_public_url: str = ""

    # Embedding dimension for the knowledge_chunks vector column. Locked at
    # migration time. 1536 covers OpenAI text-embedding-3-small (default) and
    # nomic-embed-text via Matryoshka padding/truncation.
    embedding_dim: int = 1536

    # Default RAG parameters (overridable per agent in the DB row).
    default_top_n: int = 6
    default_similarity_threshold: float = 0.25

    # Embedding provider/model. Both MCP-server and Celery worker honor
    # these. Switching requires re-embedding existing docs (their
    # embeddings are tied to the model that produced them).
    #
    # Default `native` uses fastembed with a small multilingual model
    # bundled in-process — works zero-config (no Ollama/OpenAI/etc).
    # Users can switch to ollama/openai/openai_compatible for higher
    # quality or faster inference.
    embedding_provider: str = "native"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_native_cache_dir: str = "/app/data/embedding_models"
    embedding_ollama_base_url: str = "http://ollama:11434"
    embedding_openai_base_url: str = "https://api.openai.com/v1"
    embedding_openai_api_key: str = ""

    # Where uploaded knowledge files live on disk (per-instance).
    knowledge_storage_path: str = "/app/data/agent_knowledge"
    knowledge_max_file_size_mb: int = 25

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENTS_", extra="ignore")


@lru_cache
def get_agent_settings() -> AgentSettings:
    return AgentSettings()
