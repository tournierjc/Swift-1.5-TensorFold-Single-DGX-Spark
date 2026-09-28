# What changed

<!-- The scripts, the Dockerfile, the docs. -->

# How it was verified

<!-- What you ran on which host. A rig is a claim about a machine: say what ran where. -->

- [ ] `scripts/build.sh` builds and prints `tensorfold --version` plus the branch-only import
- [ ] `scripts/pull.sh` completes (or resumes)
- [ ] `scripts/serve.sh` reaches a served model, `scripts/smoke.sh` answers
- [ ] the numbers `/health`, `/v1/models` and the timed completion reported here

# Limits

<!-- What this does not prove yet: memory peak, throughput, drafts, graphs, what was left unrun. -->
