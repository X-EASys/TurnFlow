# TurnFlow Project Page Design

## Goal

Create a lightweight public project-page shell for TurnFlow and make the canonical URL available at:

`https://x-easys.github.io/TurnFlow/`

The first release establishes the repository, visual template, navigation structure, and deployment path. Research content can be added later without changing the URL or build system.

## Repository and Hosting

- Repository: `X-EASys/TurnFlow`
- Visibility: public
- Default branch: `main`
- Hosting: GitHub Pages deployed by GitHub Actions
- Site base path: `/TurnFlow/`
- Custom domain: none

An explicit Pages workflow will deploy the repository root as a static artifact. This keeps publishing behavior versioned and avoids a framework-specific build step.

## Page Architecture

The initial site is a dependency-free single page:

- `index.html` provides semantic structure and placeholder section anchors.
- `styles.css` contains the responsive visual system.
- `.nojekyll` prevents GitHub Pages from applying Jekyll processing.
- `.github/workflows/pages.yml` validates and deploys the static files.
- `README.md` documents local preview and publishing behavior.

All internal links and assets use paths that work under the `/TurnFlow/` project-site prefix. No JavaScript or package manager is required for the initial release.

## Initial User Experience

The published page will include:

1. A compact top navigation with the TurnFlow wordmark.
2. A restrained academic-project hero containing the title and a “Coming soon” status.
3. Empty, addressable section shells for Overview, Resources, and Citation.
4. A minimal footer identifying the project.

Empty sections will remain visually unobtrusive while reserving stable anchors for future content. The layout will adapt to desktop and mobile widths and will respect reduced-motion preferences.

## Visual Direction

The template will use a neutral research-project aesthetic: generous whitespace, readable system fonts, a dark ink color, and one restrained accent color. It will not attempt to reproduce the inaccessible reference page, which currently resolves to GitHub Pages’ standard 404 response.

Accessibility requirements:

- Semantic landmarks and heading order.
- Visible keyboard focus states.
- Sufficient text/background contrast.
- Responsive navigation without horizontal overflow.
- No essential information conveyed through color alone.

## Deployment Flow

Pushes to `main` trigger the Pages workflow:

1. Check out the repository.
2. Configure GitHub Pages.
3. Upload the repository as a static Pages artifact.
4. Deploy to the `github-pages` environment.

The repository’s Pages source will be configured for GitHub Actions. The initial deployment is considered complete only after the public URL returns a successful response and displays the TurnFlow shell.

## Verification

Before publishing:

- Run a local static HTTP server and request the page.
- Check that required files, section anchors, page title, and viewport metadata exist.
- Inspect the rendered page at desktop and mobile viewport sizes.
- Confirm there are no missing local assets or browser console errors.

After publishing:

- Wait for the Pages workflow to complete successfully.
- Request `https://x-easys.github.io/TurnFlow/` and confirm an HTTP success response.
- Open the deployed page and verify the expected title, layout, and responsive behavior.

## Failure Handling

- A failed Pages run remains visible in GitHub Actions and does not silently replace the last successful deployment.
- If Pages activation is delayed, poll the workflow and Pages status before diagnosing the public URL.
- Because the site has no runtime dependencies, deployment failures are limited to repository permissions, Pages configuration, or malformed static files.

## Acceptance Criteria

- `X-EASys/TurnFlow` exists as a public repository.
- The repository contains the documented static template and deployment workflow.
- The GitHub Pages workflow completes successfully from `main`.
- `https://x-easys.github.io/TurnFlow/` loads the TurnFlow placeholder page on desktop and mobile.
- Future content can be added to the reserved sections without changing the canonical URL or deployment architecture.
