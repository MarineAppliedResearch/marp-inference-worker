# marp-inference-worker — Claude Code

See **[AGENTS.md](AGENTS.md)**. It is the single source for how to work in this
repository, whichever assistant is reading it. Shared platform conventions are synced into
the top of that file from the [umbrella repository](https://github.com/MarineAppliedResearch/MARP).

@AGENTS.md

<!-- The line above is not decoration: Claude Code imports that file here, so the
     platform rules arrive with this one rather than depending on the agent choosing to
     follow a link. A pointer is a hop, and a hop is where a hurried agent skips. -->


## Claude-specific

`.claude/settings.json` holds the permission rules; `.claude/hooks/` holds the gates.
**spec-gate** refuses edits outside `.marp/` while a `blocking` assumption in
`.marp/task.md` is unanswered; **danger-gate** stops force pushes and branch deletions.
Both fail open when the umbrella is not checked out beside this repository.

There is no test job in CI here and the branches are unresolved — see *State of this
repository* in AGENTS.md. Run `pytest` yourself and say in the verification package what
you actually ran. Do not report a tier as passing that you did not run.
