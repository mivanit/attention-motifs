"""Frontend integration tests using Playwright.

Tests that all generated HTML pages load without JavaScript errors.
The test pipeline generates all required data, so any error should fail the test.
"""

import pytest
from playwright.sync_api import ConsoleMessage, Error as PlaywrightError, Page, Response

# Pages to test with their paths relative to tests/.temp/
FRONTEND_PAGES: list[tuple[str, str]] = [
	("index.html", "Landing page"),
	("vis/attnpedia/index.html", "AttentionPedia"),
	("vis/embeds/patterns/index.html", "Pattern embeddings"),
	("vis/embeds/heads/index.html", "Head embeddings"),
	("vis/clustering/index.html", "Clustering visualization"),
	("figures/head_embed_table.html", "Head embedding table"),
	("figures/classifications.html", "Classifications page"),
]


class ConsoleErrorCollector:
	"""Collects console errors from a page."""

	def __init__(self) -> None:
		self.errors: list[str] = []
		self.warnings: list[str] = []

	def handle_console(self, message: ConsoleMessage) -> None:
		"""Handle a console message from the page."""
		text: str = message.text
		msg_type: str = message.type

		if msg_type == "error":
			self.errors.append(text)
		elif msg_type == "warning":
			self.warnings.append(text)


@pytest.mark.playwright
class TestFrontendPages:
	"""Test that all frontend pages load without errors."""

	@pytest.mark.parametrize(
		"page_path,description",
		FRONTEND_PAGES,
		ids=[p[1] for p in FRONTEND_PAGES],
	)
	def test_page_loads_without_errors(
		self,
		page: Page,
		http_server: str,
		page_path: str,
		description: str,
	) -> None:
		"""Test that a page loads without JavaScript errors.

		Args:
			page: Playwright page fixture
			http_server: Base URL of the test server
			page_path: Path to the HTML file relative to tests/.temp/
			description: Human-readable description for test output
		"""
		url: str = f"{http_server}/{page_path}"
		collector: ConsoleErrorCollector = ConsoleErrorCollector()

		# Listen for console errors
		page.on("console", collector.handle_console)

		# Track page errors (uncaught exceptions)
		page_errors: list[str] = []

		def handle_page_error(error: PlaywrightError) -> None:
			page_errors.append(str(error))

		page.on("pageerror", handle_page_error)

		# Track failed network requests (to get actual URLs for 404s)
		failed_requests: list[str] = []

		def handle_response(response: Response) -> None:
			if response.status >= 400:
				failed_requests.append(f"{response.status} {response.url}")

		page.on("response", handle_response)

		# Load the page with timeout
		try:
			page.goto(url, timeout=30000, wait_until="networkidle")
		except Exception as e:
			# Some pages may not reach networkidle due to CDN requests
			# Try with domcontentloaded instead
			if "timeout" in str(e).lower():
				page.goto(url, timeout=30000, wait_until="domcontentloaded")
			else:
				raise

		# Give Alpine.js and other frameworks time to initialize
		page.wait_for_timeout(1000)

		# Check for errors
		all_errors: list[str] = collector.errors + page_errors

		if all_errors:
			error_report: str = "\n".join(f"  - {e}" for e in all_errors)
			failed_report: str = ""
			if failed_requests:
				failed_report = "\nFailed requests:\n" + "\n".join(
					f"  - {r}" for r in failed_requests
				)
			pytest.fail(
				f"Page '{description}' ({page_path}) had {len(all_errors)} error(s):\n"
				f"{error_report}{failed_report}"
			)


@pytest.mark.playwright
class TestLandingPage:
	"""Specific tests for the landing page."""

	def test_landing_page_has_content(
		self,
		page: Page,
		http_server: str,
	) -> None:
		"""Test that landing page loads and has content."""
		url: str = f"{http_server}/index.html"

		page.goto(url, timeout=30000, wait_until="domcontentloaded")

		# Check that the page has some content
		body = page.locator("body")
		assert body.is_visible(), "Page body should be visible"


@pytest.mark.playwright
class TestAttentionPedia:
	"""Specific tests for AttentionPedia."""

	def test_attentionpedia_alpine_initializes(
		self,
		page: Page,
		http_server: str,
	) -> None:
		"""Test that AttentionPedia's Alpine.js app initializes correctly."""
		url: str = f"{http_server}/vis/attnpedia/index.html"

		page.goto(url, timeout=30000, wait_until="domcontentloaded")
		page.wait_for_timeout(2000)  # Wait for Alpine to initialize

		# Check that Alpine.js has initialized the app
		app_element = page.locator("[x-data]")
		assert app_element.count() > 0, "Alpine.js app container not found"
