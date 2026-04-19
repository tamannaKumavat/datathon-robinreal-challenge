# How Claude Helped

Built with Claude Code (Sonnet 4.6) running directly on the AWS instance via VS Code.

## What Claude Did

**Ideas & design** — Proposed separating retrieval from ranking, designed multi-turn
deduplication via a seen-listing penalty, and shaped the two-tool MCP surface after
reading the dev branch.

**Coding** — Wired VLM scores end-to-end through the API, implemented Markdown listing
cards in the MCP server, added auto-profile-update on every search turn, and fixed
config to auto-load `.env` on startup.

**Debugging** — Traced a silent stub-score bug in `/pipeline_embed`, diagnosed port 8081
being blocked at the AWS security group (not the app), and caught a `None` profile issue
for first-time users.

**AWS & MCP** — Opened port 8081 via AWS CLI, managed two concurrent services on the
instance, wrote the MCP two-step handshake instructions for teammates, and ran a 3-turn
end-to-end conversation test to validate the full pipeline.
