# econ-data-mcp

**One-stop MCP + API layer for economic data across 10+ sources**

`econ-data-mcp` makes economic data easy to access from AI tools. Query FRED, World Bank, IMF, Comtrade, and other sources in plain English through one interface.

## Two Ways To Use

### 1. Use the Website (No Setup)

- Main site: [OpenEcon.ai](https://openecon.ai)
- Live data app: [data.openecon.io/chat](https://data.openecon.io/chat)

### 2. Add MCP To Your AI Agent

- Hosted MCP endpoint: `https://data.openecon.io/mcp`

Add to Codex:
```bash
codex mcp add econ-data-mcp --url https://data.openecon.io/mcp
codex mcp get econ-data-mcp
```

Add to Claude Code:
```bash
claude mcp add --transport sse econ-data-mcp https://data.openecon.io/mcp --scope user
claude mcp get econ-data-mcp
```

## What You Can Ask

- `Use query_data to compare US, UK, and Japan inflation from 2015 to 2025.`
- `Use query_data to fetch China exports to the United States from 2020 to 2024.`
- `Use query_data to show US unemployment rate and CPI together since 2010.`
- `Use query_data to retrieve EUR/USD exchange rate history for the last 24 months.`

## Data Sources

`econ-data-mcp` unifies 10+ providers including:
- FRED
- World Bank
- UN Comtrade
- Statistics Canada
- IMF
- BIS
- Eurostat
- OECD
- ExchangeRate-API
- CoinGecko

## Developer And Contributor Docs

Technical setup, architecture, local development, testing, deployment, and contribution workflow are in:
- [Developer & Contributor Guide](docs/development/DEVELOPER_CONTRIBUTOR_GUIDE.md)
- [Documentation Index](docs/README.md)

## Support

- Issues: [GitHub Issues](https://github.com/hanlulong/econ-data-mcp/issues)
- Security: [SECURITY.md](.github/SECURITY.md)
- License: [LICENSE](LICENSE)
