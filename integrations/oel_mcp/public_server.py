from integrations.oel_mcp.protocol import OELMCPServer
from integrations.oel_mcp.public_handlers import PublicOELMCPHandlers


def main() -> None:
    OELMCPServer(PublicOELMCPHandlers(profile="public_local")).serve()


if __name__ == "__main__":
    main()
