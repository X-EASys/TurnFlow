import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"
CSS = ROOT / "styles.css"
WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"
README = ROOT / "README.md"


class ElementProbe(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []
        self.resource_nav_count = 0
        self.resource_links = []
        self._in_resource_nav = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.elements.append((tag, attributes))
        if tag == "nav" and attributes.get("aria-label") == "Project resources":
            self.resource_nav_count += 1
            self._in_resource_nav = True
        elif tag == "a" and self._in_resource_nav:
            self.resource_links.append(attributes)

    def handle_endtag(self, tag):
        if tag == "nav" and self._in_resource_nav:
            self._in_resource_nav = False


def parse_elements(html):
    probe = ElementProbe()
    probe.feed(html)
    return probe.elements


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

    def test_semantic_landmarks_and_release_sections(self):
        html = INDEX.read_text(encoding="utf-8")
        for landmark in ("header", "main", "footer"):
            self.assertRegex(html, rf"<{landmark}\b")
        for target in ("main", "abstract", "overview", "results", "citation"):
            self.assertIn(f'id="{target}"', html)
        self.assertIn('href="#main"', html)

    def test_academic_hero_is_minimal(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('id="page-title">TurnFlow</h1>', html)
        self.assertIn('class="project-type">Research project</p>', html)
        self.assertIn("Coming soon", html)
        self.assertNotIn("hero-mark", html)

    def test_only_published_resource_is_linked(self):
        html = INDEX.read_text(encoding="utf-8")
        probe = ElementProbe()
        probe.feed(html)
        self.assertEqual(probe.resource_nav_count, 1)
        self.assertEqual(len(probe.resource_links), 1)

        code_link = probe.resource_links[0]
        self.assertEqual(code_link.get("class"), "resource-link")
        self.assertEqual(code_link.get("href"), "https://github.com/X-EASys/TurnFlow")
        self.assertEqual(code_link.get("target"), "_blank")
        self.assertEqual(set(code_link.get("rel", "").split()), {"noopener", "noreferrer"})
        self.assertNotRegex(html, r'href=(?:"|\')(?:#|)(?:"|\')')

    def test_unreleased_sections_are_present_but_hidden(self):
        html = INDEX.read_text(encoding="utf-8")
        elements = parse_elements(html)
        hidden_sections = {
            attrs.get("id")
            for tag, attrs in elements
            if tag == "section" and "hidden" in attrs
        }
        self.assertEqual(
            hidden_sections,
            {"abstract", "overview", "results", "citation"},
        )

    def test_no_remote_runtime_dependencies(self):
        html = INDEX.read_text(encoding="utf-8")
        remote_asset = re.compile(
            r'<(?:script|link)\b[^>]+(?:src|href)="https?://',
            re.IGNORECASE,
        )
        self.assertIsNone(remote_asset.search(html))

    def test_local_stylesheet_resolves(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn('href="styles.css"', html)
        self.assertTrue(CSS.is_file())

    def test_accessible_responsive_css_contract(self):
        css = CSS.read_text(encoding="utf-8")
        for required_rule in (
            ":focus-visible",
            ".project-header",
            ".resource-links",
            "flex-wrap: wrap",
            "@media (max-width: 640px)",
            "@media (prefers-reduced-motion: reduce)",
            "overflow-wrap: anywhere",
            "font-size: clamp(3.25rem, 11vw, 7.5rem);",
        ):
            self.assertIn(required_rule, css)

    def test_pages_workflow_contract(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        for expected in (
            "branches: [main]",
            "contents: read",
            "pages: write",
            "id-token: write",
            "actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4",
            "actions/configure-pages@983d7736d9b0ae728b81ab479565c72886d7745b # v5",
            "actions/upload-pages-artifact@56afc609e74202658d3ffba0e8f6dda462b719fa # v3",
            "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128 # v5",
            "python3 -m unittest discover -s tests -v",
            "path: '.'",
        ):
            self.assertIn(expected, workflow)
        self.assertLess(
            workflow.index("python3 -m unittest discover -s tests -v"),
            workflow.index("actions/configure-pages@"),
        )

    def test_readme_documents_preview_and_canonical_url(self):
        readme = README.read_text(encoding="utf-8")
        self.assertIn("https://x-easys.github.io/TurnFlow/", readme)
        self.assertIn("python3 -m http.server 8000", readme)
        self.assertIn("http://localhost:8000/", readme)


if __name__ == "__main__":
    unittest.main()
