import json
import re
import unittest
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"
CSS = ROOT / "styles.css"
WORKFLOW = ROOT / ".github" / "workflows" / "pages.yml"
README = ROOT / "README.md"
PAPER_TITLE = (
    "TurnFlow: Benchmarking User-State Reasoning in Multi-Turn LLM Agents "
    "via Scene-Guided User Simulation"
)


class ElementProbe(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []
        self.resource_nav_count = 0
        self.resource_links = []
        self.links = []
        self.text_by_tag = {"title": [], "h1": []}
        self._in_resource_nav = False
        self._capturing_tag = None
        self._captured_text = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.elements.append((tag, attributes))
        if tag == "nav" and attributes.get("aria-label") == "Project resources":
            self.resource_nav_count += 1
            self._in_resource_nav = True
        elif tag == "a":
            self.links.append(attributes)
            if self._in_resource_nav:
                self.resource_links.append(attributes)
        if tag in self.text_by_tag:
            self._capturing_tag = tag
            self._captured_text = []

    def handle_data(self, data):
        if self._capturing_tag:
            self._captured_text.append(data)

    def handle_endtag(self, tag):
        if tag == "nav" and self._in_resource_nav:
            self._in_resource_nav = False
        if tag == self._capturing_tag:
            text = " ".join("".join(self._captured_text).split())
            self.text_by_tag[tag].append(text)
            self._capturing_tag = None
            self._captured_text = []


def parse_elements(html):
    probe = ElementProbe()
    probe.feed(html)
    return probe.elements


def relative_luminance(hex_color):
    channels = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return sum(
        weight * channel
        for weight, channel in zip((0.2126, 0.7152, 0.0722), linear)
    )


def contrast_ratio(first, second):
    lighter, darker = sorted(
        (relative_luminance(first), relative_luminance(second)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


class SiteContractTests(unittest.TestCase):
    def test_static_shell_files_exist(self):
        self.assertTrue(INDEX.is_file())
        self.assertTrue((ROOT / ".nojekyll").is_file())

    def test_document_metadata(self):
        html = INDEX.read_text(encoding="utf-8")
        probe = ElementProbe()
        probe.feed(html)
        self.assertRegex(html, r'<html\s+lang="en"')
        self.assertIn('<meta charset="utf-8">', html)
        self.assertIn('name="viewport"', html)
        self.assertEqual(probe.text_by_tag["title"], [PAPER_TITLE])

        descriptions = [
            attrs.get("content")
            for tag, attrs in probe.elements
            if tag == "meta" and attrs.get("name") == "description"
        ]
        self.assertEqual(len(descriptions), 1)
        self.assertIn(PAPER_TITLE, descriptions[0])

    def test_semantic_landmarks_and_release_sections(self):
        html = INDEX.read_text(encoding="utf-8")
        for landmark in ("header", "main", "footer"):
            self.assertRegex(html, rf"<{landmark}\b")
        for target in ("main", "abstract", "overview", "results", "citation"):
            self.assertIn(f'id="{target}"', html)
        self.assertIn('href="#main"', html)
        self.assertRegex(html, r'<main\s+id="main"\s+tabindex="-1">')

    def test_academic_hero_is_minimal(self):
        html = INDEX.read_text(encoding="utf-8")
        probe = ElementProbe()
        probe.feed(html)
        self.assertEqual(probe.text_by_tag["h1"], [PAPER_TITLE])
        self.assertIn('class="project-type">Research project</p>', html)
        self.assertIn("Code and dataset available", html)
        self.assertNotIn("hero-mark", html)

    def test_only_published_resource_is_linked(self):
        html = INDEX.read_text(encoding="utf-8")
        probe = ElementProbe()
        probe.feed(html)
        self.assertEqual(probe.resource_nav_count, 1)
        self.assertEqual(len(probe.resource_links), 2)

        code_link = probe.resource_links[0]
        self.assertEqual(code_link.get("class"), "resource-link")
        self.assertEqual(code_link.get("href"), "https://github.com/X-EASys/TurnFlow")
        self.assertEqual(code_link.get("target"), "_blank")
        self.assertEqual(set(code_link.get("rel", "").split()), {"noopener", "noreferrer"})
        self.assertEqual(
            probe.resource_links[1].get("href"),
            "https://huggingface.co/datasets/Buka7yo/TurnFlow",
        )
        paper_buttons = [
            attrs for tag, attrs in probe.elements
            if tag == "button" and attrs.get("data-resource") == "paper"
        ]
        self.assertEqual(len(paper_buttons), 1)
        self.assertIn("disabled", paper_buttons[0])
        self.assertNotIn("href", paper_buttons[0])
        self.assertFalse(any("arxiv.org" in link.get("href", "") for link in probe.links))
        self.assertNotRegex(html, r'href=(?:"|\')(?:#|)(?:"|\')')

    def test_released_content_is_visible_while_citation_is_pending(self):
        html = INDEX.read_text(encoding="utf-8")
        elements = parse_elements(html)
        hidden_sections = {
            attrs.get("id")
            for tag, attrs in elements
            if tag == "section" and "hidden" in attrs
        }
        self.assertTrue({"abstract", "overview", "results"}.isdisjoint(hidden_sections))
        self.assertIn("citation", hidden_sections)

    def test_leaderboard_matches_the_490_scenario_results(self):
        data = json.loads((ROOT / "assets" / "results.json").read_text())
        self.assertEqual(data["task_sizes"], [100, 79, 100, 97, 114])
        self.assertEqual(len(data["models"]), 9)
        self.assertEqual(
            [model["total_passed"] for model in data["models"]],
            [292, 288, 259, 258, 247, 245, 244, 232, 210],
        )
        self.assertEqual(data["models"][2]["name"], "DeepSeek-V4.1-Flash")
        html = INDEX.read_text(encoding="utf-8")
        for rank, model in enumerate(data["models"], 1):
            with self.subTest(model=model["name"]):
                self.assertEqual(sum(model["passed"]), model["total_passed"])
                row = re.search(
                    rf'<tr data-model="{re.escape(model["name"])}">(.*?)</tr>',
                    html, re.DOTALL,
                )
                self.assertIsNotNone(row)
                cells = re.findall(r'<(?:td|th)\b[^>]*>(.*?)</(?:td|th)>', row.group(1), re.DOTALL)
                actual = [" ".join(unescape(re.sub(r'<[^>]+>', '', cell)).split()) for cell in cells]
                expected = [
                    str(rank), model["name"], f'{model["total_passed"]}/490',
                    f'{100 * model["total_passed"] / 490:.2f}',
                    *[f'{100 * passed / total:.2f}' for passed, total in zip(model["passed"], data["task_sizes"])],
                ]
                self.assertEqual(actual, expected)

    def test_local_links_and_images_resolve(self):
        html = INDEX.read_text(encoding="utf-8")
        elements = parse_elements(html)
        ids = {attrs["id"] for _, attrs in elements if "id" in attrs}
        images = [attrs for tag, attrs in elements if tag == "img"]
        self.assertGreaterEqual(len(images), 2)
        for tag, attrs in elements:
            for key in ("href", "src"):
                value = attrs.get(key)
                if not value or value.startswith(("https://", "http://", "mailto:")):
                    continue
                if value.startswith("#"):
                    self.assertIn(value[1:], ids)
                else:
                    self.assertTrue((ROOT / value).is_file(), value)
        self.assertTrue(all(image.get("alt") for image in images))

    def test_no_remote_runtime_dependencies(self):
        html = INDEX.read_text(encoding="utf-8")
        css = CSS.read_text(encoding="utf-8")
        for tag, attrs in parse_elements(html):
            if tag == "script":
                self.assertFalse(attrs.get("src", "").startswith(("http://", "https://")))
            if tag == "link" and "stylesheet" in attrs.get("rel", "").split():
                self.assertFalse(attrs.get("href", "").startswith(("http://", "https://")))
        self.assertIsNone(
            re.search(r'(?:@import\s+[^;]*https?://|url\(\s*["\']?https?://)', css),
        )

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
        ):
            self.assertIn(required_rule, css)
        self.assertNotIn("white-space: nowrap", css)

        paper = re.search(r"--paper:\s*(#[0-9a-f]{6})", css, re.IGNORECASE)
        accent = re.search(r"--accent:\s*(#[0-9a-f]{6})", css, re.IGNORECASE)
        self.assertIsNotNone(paper)
        self.assertIsNotNone(accent)
        self.assertGreaterEqual(contrast_ratio(paper.group(1), accent.group(1)), 4.5)

    def test_title_typography_stays_close_to_release_note(self):
        css = CSS.read_text(encoding="utf-8")
        title_rule = re.search(r"(?ms)^h1\s*\{(?P<body>.*?)^\}", css)
        note_rule = re.search(r"(?ms)^\.release-note\s*\{(?P<body>.*?)^\}", css)
        self.assertIsNotNone(title_rule)
        self.assertIsNotNone(note_rule)

        clamp_pattern = re.compile(
            r"font-size:\s*clamp\("
            r"(?P<minimum>[\d.]+)rem,\s*"
            r"(?P<fluid>[\d.]+)vw,\s*"
            r"(?P<maximum>[\d.]+)rem\)"
        )
        title_size = clamp_pattern.search(title_rule.group("body"))
        note_size = clamp_pattern.search(note_rule.group("body"))
        self.assertIsNotNone(title_size)
        self.assertIsNotNone(note_size)

        def computed_size(match, viewport_width):
            minimum = float(match.group("minimum")) * 16
            fluid = float(match.group("fluid")) * viewport_width / 100
            maximum = float(match.group("maximum")) * 16
            return min(max(minimum, fluid), maximum)

        for viewport_width, minimum_ratio, maximum_ratio in (
            (320, 1.45, 1.55),
            (390, 1.45, 1.55),
            (1280, 1.85, 1.95),
            (1440, 1.85, 1.95),
        ):
            with self.subTest(viewport_width=viewport_width):
                title_value = computed_size(title_size, viewport_width)
                note_value = computed_size(note_size, viewport_width)
                ratio = title_value / note_value
                self.assertGreaterEqual(ratio, minimum_ratio)
                self.assertLessEqual(ratio, maximum_ratio)

        weight = re.search(r"font-weight:\s*(\d+)", title_rule.group("body"))
        line_height = re.search(r"line-height:\s*([\d.]+)", title_rule.group("body"))
        self.assertIsNotNone(weight)
        self.assertIsNotNone(line_height)
        self.assertEqual(int(weight.group(1)), 680)
        self.assertEqual(float(line_height.group(1)), 1.12)

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
            "python3 -m unittest discover -s tests -p test_site.py -v",
            "path: '_site'",
            "cp -R assets _site/assets",
        ):
            self.assertIn(expected, workflow)
        self.assertLess(
            workflow.index("python3 -m unittest discover -s tests -p test_site.py -v"),
            workflow.index("actions/configure-pages@"),
        )

    def test_readme_documents_preview_and_canonical_url(self):
        readme = README.read_text(encoding="utf-8")
        self.assertIn("https://x-easys.github.io/TurnFlow/", readme)
        self.assertIn("python3 -m http.server 8000", readme)
        self.assertIn("http://localhost:8000/", readme)


if __name__ == "__main__":
    unittest.main()
