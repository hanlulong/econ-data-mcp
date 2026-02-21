<p align="center">
  <img src="packages/frontend/public/favicon.svg" width="80" height="80" alt="OpenEcon logo" />
</p>

<h1 align="center">OpenEcon</h1>

<p align="center">
  <strong>Query economic data from 10+ sources using plain English.</strong><br/>
  One natural-language interface for FRED, World Bank, IMF, Eurostat, Comtrade, and more.
</p>

<p align="center">
  <a href="https://openecon.ai"><img src="https://img.shields.io/badge/Live_Demo-openecon.ai-blue?style=flat-square" alt="Live Demo" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="MIT License" /></a>
  <a href="https://github.com/hanlulong/econ-data-mcp/issues"><img src="https://img.shields.io/github/issues/hanlulong/econ-data-mcp?style=flat-square" alt="Issues" /></a>
  <a href="https://github.com/hanlulong/econ-data-mcp/stargazers"><img src="https://img.shields.io/github/stars/hanlulong/econ-data-mcp?style=flat-square" alt="Stars" /></a>
</p>

<p align="center">
  <a href="https://openecon.ai">Website</a> &middot;
  <a href="https://data.openecon.io/chat">Live App</a> &middot;
  <a href="docs/README.md">Docs</a> &middot;
  <a href="docs/development/DEVELOPER_CONTRIBUTOR_GUIDE.md">Contributing</a>
</p>

---

<!-- TODO: Replace with an actual GIF/screenshot of the app in action.
     Record a 15-second GIF: type a query → see the chart appear.
     Save as docs/assets/demo.gif and uncomment:
<p align="center">
  <img src="docs/assets/demo.gif" width="720" alt="OpenEcon demo — ask a question, get a chart" />
</p>
-->

## Why OpenEcon?

Getting economic data today means juggling APIs, reading docs for each provider, wrangling country codes, and normalizing date formats. OpenEcon fixes this:

- **Ask in English** — "Compare US and Japan inflation since 2015" just works
- **One interface, 10+ sources** — FRED, World Bank, IMF, Eurostat, BIS, UN Comtrade, and more, all unified
- **MCP-native** — plug into Claude, Codex, or any MCP-compatible AI agent with one command
- **Self-hostable** — MIT-licensed, run locally or deploy your own instance
- **330,000+ indicators** — full-text search across the world's major economic databases

## Quick Start

### Use the hosted app (no setup)

Try it now at **[data.openecon.io/chat](https://data.openecon.io/chat)** — no account required.

### Add to your AI agent (one command)

**Claude Code:**
```bash
claude mcp add --transport sse econ-data-mcp https://data.openecon.io/mcp --scope user
```

**Codex:**
```bash
codex mcp add econ-data-mcp --url https://data.openecon.io/mcp
```

Then ask your agent:
```
Use query_data to compare US, UK, and Japan inflation from 2015 to 2025.
```

### Self-host

```bash
git clone https://github.com/hanlulong/econ-data-mcp.git
cd econ-data-mcp
cp .env.example .env          # Add your OPENROUTER_API_KEY
pip install -r requirements.txt
npm install
python3 scripts/restart_dev.py
# Backend: http://localhost:3001  |  Frontend: http://localhost:5173
```

## Example Queries

| Query | Sources Used |
|-------|-------------|
| "US GDP growth for the last 10 years" | FRED |
| "Compare China, India, and Brazil GDP growth 2018–2024" | World Bank |
| "EUR/USD exchange rate history last 24 months" | ExchangeRate-API |
| "US unemployment rate and CPI together since 2010" | FRED |
| "China exports to the United States 2020–2024" | UN Comtrade |
| "EU debt-to-GDP ratios across member states" | Eurostat |
| "Bitcoin price history for the last year" | CoinGecko |

## Features

**Natural Language Interface** — Ask questions in plain English. An LLM parses your intent, picks the right provider, and fetches the data.

**Smart Indicator Discovery** — 330K+ indicators indexed with full-text search. No need to know series codes — just describe what you want.

**Streaming Results** — Real-time progress via Server-Sent Events. See each step as it happens: parsing, routing, fetching, charting.

**Pro Mode** — For complex analysis, the system generates and executes Python code in a sandboxed environment, producing publication-ready visualizations.

**MCP Server** — First-class Model Context Protocol support. Any MCP-compatible AI agent can query economic data through the hosted endpoint.

**Self-Hostable & Extensible** — MIT-licensed. Add new providers by implementing a single base class. Full plugin architecture.

## Data Sources

| Provider | Coverage | Indicators | API Key |
|----------|----------|-----------|---------|
| **FRED** | US macroeconomic data | 90,000+ series | Free |
| **World Bank** | Global development | 16,000+ indicators | None |
| **IMF** | International financial statistics | Extensive | None |
| **Eurostat** | EU member states | Extensive | None |
| **UN Comtrade** | International trade flows | All HS codes | Free |
| **BIS** | Central bank & financial stability | Curated | None |
| **Statistics Canada** | Canadian economic data | 40,000+ tables | None |
| **OECD** | OECD member countries | Extensive | None |
| **ExchangeRate-API** | 160+ currencies | Live & historical | Free |
| **CoinGecko** | Cryptocurrencies | 10,000+ coins | Free |

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
│  User / Agent   │────▶│  FastAPI Backend  │────▶│  Data Providers          │
│                 │     │                  │     │                          │
│  "US inflation" │     │  LLM Parser      │     │  FRED · World Bank · IMF │
│                 │◀────│  Query Router    │◀────│  Eurostat · BIS · ...    │
│  📊 Chart + Data │     │  Normalizer      │     │                          │
└─────────────────┘     └──────────────────┘     └──────────────────────────┘
        │                        │
   React Frontend          MCP Endpoint
   (Vite + Recharts)     (SSE Transport)
```

**Stack:** Python · FastAPI · React · TypeScript · Vite · Recharts · FAISS · OpenRouter

## Pro Mode Examples

AI-generated analysis with Python code execution:

<p align="center">
  <img src="public_media/promode/promode_18a6ff9c_phillips_curve.png" width="380" alt="Phillips Curve analysis" />
  <img src="public_media/promode/promode_299c8cdd_gdp_growth_top10_timeseries.png" width="380" alt="GDP growth top 10 economies" />
</p>
<p align="center">
  <img src="public_media/promode/promode_05a396dc_eu_unemployment_timeseries.png" width="380" alt="EU unemployment trends" />
  <img src="public_media/promode/promode_23f16b4f_renewable_energy_investment_by_region.png" width="380" alt="Renewable energy investment" />
</p>

## Contributing

We welcome contributions! See the [Developer & Contributor Guide](docs/development/DEVELOPER_CONTRIBUTOR_GUIDE.md) for setup instructions, architecture overview, and code standards.

**Quick links:**
- [Open issues](https://github.com/hanlulong/econ-data-mcp/issues) — bug reports and feature requests
- [Documentation](docs/README.md) — full docs index
- [Security policy](.github/SECURITY.md) — responsible disclosure

## License

[MIT](LICENSE) — use it however you want.
