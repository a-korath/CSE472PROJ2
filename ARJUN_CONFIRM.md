# Arjun: one thing to confirm before we submit (Mon Nov 23)

The report is final and verified (see `report/VERIFICATION.md`). The only item that needs you is the **AI-use statement**
in Acknowledgments (`report/main.tex`, near the end):

> **Use of generative AI:** we used Anthropic's Claude (Claude Opus models, via the Claude app and Claude Code) to help
> understand the papers, plan the work, write and debug code and job scripts, analyse results and draft and edit this
> report. All design decisions, experiments and claims were checked by the authors, and every number in this report is
> generated from our run files.

**What to do**
- If this covers every AI tool you used for this project (ChatGPT, Copilot, Gemini, Cursor and so on count too): reply
  "confirmed" to Aryan. Nothing else to do.
- If you used another tool: add it to that sentence (e.g. "…and GitHub Copilot for code completion"), run
  `cd report && latexmk -pdf main.tex`, check it is still 8 pages, then commit and push. Or just tell Aryan the tool name
  and he will add it.

**Also decide with Aryan:** who uploads to Canvas (group or individual submission). Open the uploaded file afterwards to check it.
