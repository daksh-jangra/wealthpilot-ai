# ADR 001: CrewAI for Multi-Agent Orchestration

**Status:** Accepted  
**Date:** 2025-03-01

## Context

The rebalancing pipeline requires six specialised agents to collaborate on each decision. We evaluated LangGraph, AutoGen, and CrewAI.

## Decision

Use **CrewAI** with a hierarchical process (Orchestrator → specialist crew).

## Rationale

- Native support for hierarchical manager/worker patterns
- Built-in task dependency chain: each agent's output is the next agent's input
- Role-based backstory system matches our domain expert model
- Retry and delegation mechanism handles compliance rejection loops
- LangChain-compatible: can use Claude via `langchain-anthropic`

## Consequences

- Requires `crewai` and `langchain-anthropic` dependencies
- Agent communication is sequential within a crew (not fully parallel)
- CrewAI API changes between versions — pin to `>=0.80.0`
