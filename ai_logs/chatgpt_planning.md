# ChatGPT planning and prompt refinement

I used ChatGPT as a planning and prompt-refinement assistant before sending
instructions to Codex. ChatGPT helped analyze the assignment, clarify the
roadmap, identify relevant metrics, and formulate Codex prompts.

## How this is used in practice

- Analyze the assignment brief and turn it into a milestone-by-milestone roadmap (M0-M4), plus an M5 few-step stretch added on top.
- Clarify priorities (correctness and explainability over sophistication, mode coverage as the central concern).
- Choose sample-based metrics suited to a multimodal 2D target.
- Draft and refine the prompts that are then handed to Codex for implementation.

M0-M4 were implemented via ChatGPT (planning) -> Codex (coding). The M5 few-step
DMD2 stretch was implemented in the Cursor agent (Claude); see `index.md` for the
per-session breakdown. Codex exports are stored under `transcripts/`.