# TurnFlow Project Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Publish a lightweight, responsive TurnFlow project-page shell at https://x-easys.github.io/TurnFlow/.

**Architecture:** Build a dependency-free single-page site from semantic HTML and a focused CSS stylesheet. Validate the site contract with Python standard-library tests, then deploy the repository root through GitHub’s official Pages Actions workflow.

**Tech Stack:** HTML5, CSS3, Python 3 unittest, GitHub Actions, GitHub Pages

**Spec:** docs/superpowers/specs/2026-08-30-turnflow-project-page-design.md

## Global Constraints

- Repository: X-EASys/TurnFlow.
- Visibility: public.
- Default branch: main.
- Canonical URL: https://x-easys.github.io/TurnFlow/.
- Site base path: /TurnFlow/.
- Custom domain: none.
- The initial release has no JavaScript, package manager, framework, remote font, or runtime dependency.
- The page must expose stable overview, resources, and citation anchors.
- The page must work at desktop and mobile widths, retain visible keyboard focus, and respect reduced-motion preferences.

## File Map

- index.html: semantic page structure, navigation, hero, empty section shells, and footer.
- styles.css: visual tokens, layout, responsive behavior, focus treatment, and reduced-motion behavior.
- .nojekyll: disables Jekyll processing for the static artifact.
- tests/test_site.py: executable contract for markup, local assets, CSS safeguards, workflow, and README.
- .github/workflows/pages.yml: official static Pages deployment flow.
- README.md: canonical URL, local preview instructions, and publishing behavior.

---

### Task 1: Semantic Static Page Shell

**Files:**

- Create: tests/test_site.py
- Create: index.html
- Create: .nojekyll

**Interfaces:**

- Consumes: the repository root and the approved design specification.
- Produces: stable DOM IDs main, overview, resources, and citation; a local stylesheet reference to styles.css; a dependency-free HTML document for Task 2.

- [ ] **Step 1: Write the failing markup contract**

Create tests/test_site.py with:

~~~python
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"
CSS = ROOT / "styles.css"
WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"
README = ROOT / "README.md"


class SiteContractTests(unittest.TestCase):
    def test_static_shell_files_exist(self):
        self.assertTrue(INDEX.is_file())
        self.assertTrue((ROOT / ".nojekyll").is_file())

    def test_document_metadata(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertRegex(html, r'<html\s+lang="en"')
        self.assertIn('<meta charset="utf-8">', html)
        self.assertIn('name="viewport"', html)
        self.assertIn("<title>TurnFlow</title>", html)
        self.assertIn('name="description"', html)

    def test_semantic_landmarks_and_stable_anchors(self):
        html = INDEX.read_text(encoding="utf-8")
        for landmark in ("header", "nav", "main", "footer"):
            self.assertRegex(html, rf"<{landmark}\b")
        for target in ("main", "overview", "resources", "citation"):
            self.assertIn(f'id="{target}"', html)
        for href in ("#main", "#overview", "#resources", "#citation"):
            self.assertIn(f'href="{href}"', html)

    def test_initial_copy_is_minimal(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('id="page-title">TurnFlow</h1>', html)
        self.assertIn("Coming soon", html)

    def test_no_remote_runtime_dependencies(self):
        html = INDEX.read_text(encoding="utf-8")
        remote_asset = re.compile(
            r'<(?:script|link)\b[^>]+(?:src|href)="https?://',
            re.IGNORECASE,
        )
        self.assertIsNone(remote_asset.search(html))


if __name__ == "__main__":
    unittest.main()
~~~

- [ ] **Step 2: Run the contract and confirm the expected failure**

Run:

~~~bash
python3 -m unittest tests/test_site.py -v
~~~

Expected: FAIL because index.html and .nojekyll do not exist.

- [ ] **Step 3: Implement the semantic page shell**

Create index.html with:

~~~html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="TurnFlow project page.">
    <meta name="theme-color" content="#f4f7f3">
    <title>TurnFlow</title>
    <link rel="stylesheet" href="styles.css">
  </head>
  <body>
    <a class="skip-link" href="#main">Skip to content</a>

    <header class="site-header">
      <nav class="nav shell" aria-label="Primary navigation">
        <a class="brand" href="./" aria-label="TurnFlow home">
          Turn<span class="brand-accent">Flow</span>
        </a>
        <ul class="nav-list">
          <li><a href="#overview">Overview</a></li>
          <li><a href="#resources">Resources</a></li>
          <li><a href="#citation">Citation</a></li>
        </ul>
      </nav>
    </header>

    <main id="main">
      <section class="hero shell" aria-labelledby="page-title">
        <div class="hero-copy">
          <p class="eyebrow">Research project</p>
          <h1 id="page-title">TurnFlow</h1>
          <p class="status"><span class="status-dot" aria-hidden="true"></span>Coming soon</p>
        </div>
        <div class="hero-mark" aria-hidden="true">
          <span></span><span></span><span></span>
        </div>
      </section>

      <div class="sections shell">
        <section id="overview" class="section-card" aria-labelledby="overview-title">
          <h2 id="overview-title">Overview</h2>
          <div class="section-space" aria-hidden="true"></div>
        </section>
        <section id="resources" class="section-card" aria-labelledby="resources-title">
          <h2 id="resources-title">Resources</h2>
          <div class="section-space" aria-hidden="true"></div>
        </section>
        <section id="citation" class="section-card" aria-labelledby="citation-title">
          <h2 id="citation-title">Citation</h2>
          <div class="section-space" aria-hidden="true"></div>
        </section>
      </div>
    </main>

    <footer class="site-footer">
      <div class="shell"><p>TurnFlow</p></div>
    </footer>
  </body>
</html>
~~~

Create an empty .nojekyll file.

- [ ] **Step 4: Run the markup contract and confirm it passes**

Run:

~~~bash
python3 -m unittest tests/test_site.py -v
~~~

Expected: 5 tests pass.

- [ ] **Step 5: Commit the semantic shell**

~~~bash
git add .nojekyll index.html tests/test_site.py
git commit -m "feat: add TurnFlow page shell"
~~~

---

### Task 2: Responsive Visual Template

**Files:**

- Modify: tests/test_site.py
- Create: styles.css

**Interfaces:**

- Consumes: the class names and styles.css link defined by Task 1.
- Produces: a complete local stylesheet with responsive, keyboard-focus, and reduced-motion behavior.

- [ ] **Step 1: Add failing stylesheet safeguards**

Add these methods to SiteContractTests before the module’s final conditional:

~~~python
    def test_local_stylesheet_resolves(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('href="styles.css"', html)
        self.assertTrue(CSS.is_file())

    def test_accessible_responsive_css_contract(self):
        css = CSS.read_text(encoding="utf-8")
        for required_rule in (
            ":focus-visible",
            "@media (max-width: 680px)",
            "@media (prefers-reduced-motion: reduce)",
            "overflow-wrap: anywhere",
        ):
            self.assertIn(required_rule, css)
~~~

- [ ] **Step 2: Run the stylesheet safeguards and confirm failure**

Run:

~~~bash
python3 -m unittest tests.test_site.SiteContractTests.test_local_stylesheet_resolves tests.test_site.SiteContractTests.test_accessible_responsive_css_contract -v
~~~

Expected: ERROR or FAIL because styles.css does not exist.

- [ ] **Step 3: Implement the responsive visual system**

Create styles.css with:

~~~css
:root {
  color-scheme: light;
  --paper: #f4f7f3;
  --surface: #ffffff;
  --ink: #17201d;
  --muted: #68736e;
  --line: #dce4df;
  --accent: #187a68;
  --accent-soft: #dcefe9;
  --shadow: 0 20px 55px rgb(23 32 29 / 8%);
  --radius: 22px;
  --shell: min(1120px, calc(100% - 48px));
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
  scroll-padding-top: 88px;
}

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.6;
}

a {
  color: inherit;
}

:focus-visible {
  outline: 3px solid var(--accent);
  outline-offset: 4px;
}

.skip-link {
  position: fixed;
  z-index: 100;
  top: 12px;
  left: 12px;
  padding: 10px 14px;
  border-radius: 10px;
  background: var(--ink);
  color: white;
  transform: translateY(-160%);
  transition: transform 160ms ease;
}

.skip-link:focus {
  transform: translateY(0);
}

.shell {
  width: var(--shell);
  margin-inline: auto;
}

.site-header {
  position: sticky;
  z-index: 20;
  top: 0;
  border-bottom: 1px solid rgb(220 228 223 / 80%);
  background: rgb(244 247 243 / 88%);
  backdrop-filter: blur(16px);
}

.nav {
  min-height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 32px;
}

.brand {
  font-size: 1.05rem;
  font-weight: 760;
  letter-spacing: -0.03em;
  text-decoration: none;
}

.brand-accent {
  color: var(--accent);
}

.nav-list {
  display: flex;
  align-items: center;
  gap: 24px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.nav-list a {
  color: var(--muted);
  font-size: 0.9rem;
  font-weight: 650;
  text-decoration: none;
}

.nav-list a:hover {
  color: var(--ink);
}

.hero {
  min-height: 68vh;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(260px, 0.55fr);
  align-items: center;
  gap: clamp(48px, 8vw, 112px);
  padding-block: clamp(80px, 12vw, 150px);
}

.eyebrow {
  margin: 0 0 16px;
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 760;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

h1,
h2,
p {
  overflow-wrap: anywhere;
}

h1 {
  max-width: 10ch;
  margin: 0;
  font-size: clamp(4.5rem, 13vw, 9rem);
  font-weight: 780;
  letter-spacing: -0.075em;
  line-height: 0.88;
}

.status {
  width: fit-content;
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 32px 0 0;
  padding: 9px 14px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: rgb(255 255 255 / 68%);
  color: var(--muted);
  font-size: 0.88rem;
  font-weight: 650;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 5px var(--accent-soft);
}

.hero-mark {
  aspect-ratio: 1;
  position: relative;
  border: 1px solid var(--line);
  border-radius: 50%;
  background: linear-gradient(145deg, rgb(255 255 255 / 82%), rgb(220 239 233 / 62%));
  box-shadow: var(--shadow);
}

.hero-mark::before,
.hero-mark::after,
.hero-mark span {
  content: "";
  position: absolute;
  border: 1px solid rgb(24 122 104 / 34%);
  border-radius: 50%;
}

.hero-mark::before {
  inset: 18%;
}

.hero-mark::after {
  inset: 36%;
  background: var(--accent);
  box-shadow: 0 0 55px rgb(24 122 104 / 28%);
}

.hero-mark span:nth-child(1) {
  inset: 8% 46%;
}

.hero-mark span:nth-child(2) {
  inset: 46% 8%;
}

.hero-mark span:nth-child(3) {
  inset: 27%;
  border-style: dashed;
  transform: rotate(28deg);
}

.sections {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 20px;
  padding-bottom: clamp(80px, 12vw, 144px);
}

.section-card {
  min-height: 220px;
  padding: 28px;
  scroll-margin-top: 96px;
  border: 1px solid var(--line);
  border-radius: var(--radius);
  background: rgb(255 255 255 / 60%);
}

.section-card h2 {
  margin: 0;
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.section-space {
  min-height: 120px;
}

.site-footer {
  border-top: 1px solid var(--line);
}

.site-footer p {
  margin: 0;
  padding-block: 28px;
  color: var(--muted);
  font-size: 0.85rem;
}

@media (max-width: 800px) {
  .hero {
    grid-template-columns: 1fr;
  }

  .hero-mark {
    width: min(72vw, 360px);
  }

  .sections {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 680px) {
  :root {
    --shell: min(100% - 32px, 1120px);
  }

  .nav {
    min-height: 64px;
    gap: 16px;
  }

  .nav-list {
    gap: 12px;
  }

  .nav-list a {
    font-size: 0.78rem;
  }

  .hero {
    min-height: auto;
    padding-block: 72px 88px;
  }

  h1 {
    font-size: clamp(4rem, 24vw, 6.5rem);
  }
}

@media (prefers-reduced-motion: reduce) {
  html {
    scroll-behavior: auto;
  }

  *,
  *::before,
  *::after {
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
}
~~~

- [ ] **Step 4: Run all local contract tests**

Run:

~~~bash
python3 -m unittest discover -s tests -v
~~~

Expected: 7 tests pass.

- [ ] **Step 5: Preview and inspect both target viewport classes**

Run:

```bash
python3 -m http.server 8000
```

Open http://localhost:8000/ and inspect at approximately 1440×900 and 390×844. Confirm the navigation stays visible, no horizontal scrolling occurs, the hero remains balanced, the three cards stack on mobile, and keyboard Tab reveals the skip link and visible focus rings. Stop the server after inspection.

- [ ] **Step 6: Commit the visual template**

~~~bash
git add styles.css tests/test_site.py
git commit -m "feat: style responsive TurnFlow template"
~~~

---

### Task 3: Pages Workflow and Project Documentation

**Files:**

- Modify: tests/test_site.py
- Create: .github/workflows/pages.yml
- Create: README.md

**Interfaces:**

- Consumes: the tested static files from Tasks 1 and 2 and the main branch.
- Produces: a Pages workflow that publishes the repository root and operator documentation for local preview and release verification.

- [ ] **Step 1: Add failing deployment and documentation tests**

Add these methods to SiteContractTests before the module’s final conditional:

~~~python
    def test_pages_workflow_contract(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        for expected in (
            "branches: [main]",
            "contents: read",
            "pages: write",
            "id-token: write",
            "actions/checkout@v4",
            "actions/configure-pages@v5",
            "actions/upload-pages-artifact@v3",
            "actions/deploy-pages@v5",
            "path: '.'",
        ):
            self.assertIn(expected, workflow)

    def test_readme_documents_preview_and_canonical_url(self):
        readme = README.read_text(encoding="utf-8")
        self.assertIn("https://x-easys.github.io/TurnFlow/", readme)
        self.assertIn("python3 -m http.server 8000", readme)
        self.assertIn("http://localhost:8000/", readme)
~~~

- [ ] **Step 2: Run the new tests and confirm failure**

Run:

~~~bash
python3 -m unittest tests.test_site.SiteContractTests.test_pages_workflow_contract tests.test_site.SiteContractTests.test_readme_documents_preview_and_canonical_url -v
~~~

Expected: ERROR because the workflow and README do not exist.

- [ ] **Step 3: Add the official static Pages workflow**

Create .github/workflows/pages.yml with:

~~~yaml
name: Deploy static content to Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Pages
        uses: actions/configure-pages@v5
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: '.'
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v5
~~~

- [ ] **Step 4: Document the project and release path**

Create README.md with:

~~~~markdown
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
~~~~

- [ ] **Step 5: Run the complete contract suite**

Run:

~~~bash
python3 -m unittest discover -s tests -v
~~~

Expected: 9 tests pass.

- [ ] **Step 6: Enable Pages with the workflow build type**

Run:

~~~bash
gh api --method POST repos/X-EASys/TurnFlow/pages -f build_type=workflow
~~~

Expected: a Pages-site response whose html_url is https://x-easys.github.io/TurnFlow/ and whose build_type is workflow. If the API reports that Pages already exists, run gh api repos/X-EASys/TurnFlow/pages and verify those same fields instead of attempting to recreate it.

- [ ] **Step 7: Commit and push the release configuration**

~~~bash
git add .github/workflows/pages.yml README.md tests/test_site.py
git commit -m "ci: publish TurnFlow with GitHub Pages"
git push origin main
~~~

- [ ] **Step 8: Wait for the deployment workflow**

Run:

~~~bash
run_id=$(gh run list --repo X-EASys/TurnFlow --workflow pages.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$run_id" --repo X-EASys/TurnFlow --exit-status
~~~

Expected: the deployment job completes successfully.

- [ ] **Step 9: Verify the public endpoint and rendered page**

Run:

~~~bash
curl --fail --location --silent --show-error https://x-easys.github.io/TurnFlow/ | grep -F '<title>TurnFlow</title>'
~~~

Expected: the TurnFlow title is printed and the command exits successfully.

Open https://x-easys.github.io/TurnFlow/ at desktop and mobile viewport sizes. Confirm that the published version matches the local visual checks and that browser developer tools show no missing assets or console errors.

---

### Task 4: Final Integrated Verification

**Files:**

- Verify only; no file changes expected.

**Interfaces:**

- Consumes: the published main branch, GitHub Actions run, and public Pages URL.
- Produces: final evidence that the committed source, automated contract, deployment, and public render agree.

- [ ] **Step 1: Confirm the local checkout is clean and synchronized**

Run:

~~~bash
git status --short --branch
git rev-parse HEAD
git ls-remote origin refs/heads/main
~~~

Expected: no uncommitted paths, main tracks origin/main, and the local and remote commit hashes match.

- [ ] **Step 2: Re-run the complete local contract**

Run:

~~~bash
python3 -m unittest discover -s tests -v
~~~

Expected: 9 tests pass with no failures or errors.

- [ ] **Step 3: Confirm Pages configuration and latest workflow state**

Run:

~~~bash
gh api repos/X-EASys/TurnFlow/pages --jq '{html_url, status, build_type}'
gh run list --repo X-EASys/TurnFlow --workflow pages.yml --limit 1 --json status,conclusion,url,headSha
~~~

Expected: Pages reports the canonical URL and build_type workflow; the latest run reports completed and success for the same commit pushed to main.

- [ ] **Step 4: Confirm the public response**

Run:

~~~bash
curl --fail --location --silent --show-error https://x-easys.github.io/TurnFlow/ | grep -F 'id="page-title">TurnFlow</h1>'
~~~

Expected: the TurnFlow heading is printed and the command exits successfully.
