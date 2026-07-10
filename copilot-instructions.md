# Copilot Instructions for MARP Inference Worker

## Load Project Handoff Guide

When starting any conversation about this project, load the **AGENTS.md** file at the root of this repository. It contains:

- Complete project architecture and design decisions
- Current implementation status and API endpoints
- Code style rules and working style with Isaac
- Testing preferences and patterns
- Patch delivery rules for LLMs
- Git commit guidelines

The AGENTS.md file is the authoritative source of truth for LLM assistants working on this project.

## Quick Reference

**Repository**: MarineAppliedResearch/marp-inference-worker  
**Root**: `C:\Users\isaac\Documents\Workspace\marp-inference-worker` (Windows) or equivalent on Linux/Mac  
**Language**: Python 3.10.x  
**Framework**: FastAPI  
**Testing**: pytest

## Before You Start

1. Read AGENTS.md to understand the project architecture and current state
2. Check the "Current Best Next Steps" section in AGENTS.md for Isaac's priorities
3. Follow the "Working Style with Isaac" section—work interactively, one checkpoint at a time
4. Follow the "Code Comment Style Rules" and "Patch Delivery Rules for LLMs" sections exactly

## Cross-Platform Note

This project must work on both **Windows and Linux (Ubuntu)**. When suggesting code or shell commands, provide platform-specific variants or use cross-platform approaches.

## Important

- Isaac is the programmer. You are the assistant. Ask for clarification, don't assume.
- Work small. Make precise changes. Avoid speculative rewrites.
- Run pytest after changes unless Isaac says otherwise.
