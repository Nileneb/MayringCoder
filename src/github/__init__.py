"""GitHub data-source integration.

Parses user-supplied GitHub references (URLs, slugs) into a normalised
form that other pipelines (ingest, analysis, duel) consume as a single
data source — instead of each tab inventing its own parsing.
"""
from src.github.input import (
    GitHubInput,
    GitHubInputError,
    parse_github_input,
)

__all__ = ["GitHubInput", "GitHubInputError", "parse_github_input"]
