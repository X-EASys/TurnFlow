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


if __name__ == "__main__":
    unittest.main()
