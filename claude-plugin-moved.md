# MayringCoder — Claude Plugin (Extracted)

The Claude plugin has been extracted to a separate repository for better modularity and independent versioning.

## Installation

Install the plugin from the new repository:

```bash
/plugin marketplace add Nileneb/mayring-claude-plugin
/plugin install mayring-claude-plugin
```

Or clone directly for development:

```bash
git clone https://github.com/Nileneb/mayring-claude-plugin.git
cd mayring-claude-plugin
bash install.sh
```

## Repository

**New home:** [Nileneb/mayring-claude-plugin](https://github.com/Nileneb/mayring-claude-plugin)

This directory previously contained the plugin code but has been extracted to improve:
- Independent versioning (plugin ≠ API version)
- Cleaner workspace for backend-only development
- Reduced noise for MayringCoder contributors

## What's Included

The extracted plugin provides:
- Memory integration hooks (UserPromptSubmit, Stop)
- Pi-Agent tools for async code generation
- MCP server for local development
- Custom Skills for specialized tasks

See the [mayring-claude-plugin repository](https://github.com/Nileneb/mayring-claude-plugin/blob/main/README.md) for full documentation.
