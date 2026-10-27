# Course formula cache

The editable sources remain the LaTeX in teaching_2026_SM.html (data-tex and inline delimiters). The block between SM-PRECOMPILED-MATH:BEGIN/END is generated output; do not edit it manually.

build-math-cache.cjs renders the currently visible stories and projection pages using the repository's MathJax 3.2.2, deduplicates formula SVGs and glyph paths, and embeds the results in the HTML. Cached formulas require no math-engine or font download beyond the page's own inline bootstrap. MathJax remains an on-demand fallback for new formulas not yet in the cache.

After changing formulas, rerun from the repository root:

    node teaching/2026F_SM/tools/build-math-cache.cjs

Requirements: Node.js, Playwright and a Playwright Chromium installation. SM_PLAYWRIGHT_MODULE can point to an existing Playwright module; SM_BROWSER can point to an installed Chromium browser. For the current author's bundled runtime:

    SM_PLAYWRIGHT_MODULE=/Users/wqr/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright \
    SM_BROWSER='/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge' \
    /opt/homebrew/bin/node teaching/2026F_SM/tools/build-math-cache.cjs

The builder refuses to overwrite the HTML if it changes during compilation. It rebuilds from LaTeX rather than trusting an old cache. Future stories outside the current visibility range use the live fallback until the cache is rebuilt after making them visible. Keep the MathJax vendor directory.

Verified build on 2026-10-26: 1,463 formula occurrences, 862 unique formula/display combinations and 251 shared glyph paths. Desktop/mobile reading checks and all 15 Einstein/Planck projection pages passed. All original story text, timetable, projection JSON and LaTeX were preserved. Network checks passed with the math engine completely blocked, a cold connection throttled to 64 KiB/s with 800 ms latency, and a local file opened without network access. All 1,463 formulas rendered with zero engine requests, unresolved sources, missing glyphs or duplicate IDs. A newly inserted uncached formula also rendered after its first engine request was deliberately aborted and retried. These results verify formulas, not offline availability of unrelated remote media or fonts.
