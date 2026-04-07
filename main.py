import asyncio
import sys
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from core.claude import Claude, use_vertex_backend

from core.cli_chat import CliChat
from core.cli import CliApp

load_dotenv()

# Anthropic / Vertex (Vertex: gcloud auth application-default login)
claude_model = os.getenv("CLAUDE_MODEL", "")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")

assert claude_model, "Error: CLAUDE_MODEL cannot be empty. Update .env"

if use_vertex_backend():
    assert os.getenv("ANTHROPIC_VERTEX_PROJECT_ID"), (
        "Error: ANTHROPIC_VERTEX_PROJECT_ID is required when using Vertex "
        "(ANTHROPIC_USE_VERTEX=1 or CLAUDE_CODE_USE_VERTEX=1). Update .env or your shell."
    )
    assert os.getenv("CLOUD_ML_REGION"), (
        "Error: CLOUD_ML_REGION is required when using Vertex. Update .env or your shell."
    )
else:
    assert anthropic_api_key, (
        "Error: ANTHROPIC_API_KEY cannot be empty when Vertex is disabled. Update .env"
    )


async def main():
    claude_service = Claude(model=claude_model)

    server_scripts = sys.argv[1:]
    clients = {}

    command, args = (
        ("uv", ["run", "mcp_server.py"])
        if os.getenv("USE_UV", "0") == "1"
        else ("python", ["mcp_server.py"])
    )

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(command=command, args=args)
        )
        clients["doc_client"] = doc_client

        for i, server_script in enumerate(server_scripts):
            client_id = f"client_{i}_{server_script}"
            client = await stack.enter_async_context(
                MCPClient(command="uv", args=["run", server_script])
            )
            clients[client_id] = client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            claude_service=claude_service,
        )

        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
