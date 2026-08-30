# TurnFlow

Official project page for TurnFlow.

**Website:** https://x-easys.github.io/TurnFlow/

## Local preview

From the repository root, run:

```bash
python3 -m http.server 8000
```

Then open http://localhost:8000/.

## Publishing

Pushes to main deploy the repository root through .github/workflows/pages.yml. GitHub Pages must use the GitHub Actions build type.

## Release content

The academic project-page sections for the abstract, overview, results, and citation already exist in `index.html`, but remain hidden until the corresponding material is final. Remove a section's `hidden` attribute only when its content is ready, and add resource buttons only for working public URLs.
