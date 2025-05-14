from typing import Literal
from pathlib import Path

# TODO: move this to muutils
def inline_html_assets(
	html: str,
	assets: list[tuple[Literal["script", "style"], str]],
	base_path: Path,
) -> str:
	"""Inline specified local CSS/JS files into an HTML document.

	Each entry in `assets` should be a tuple like `("script", "app.js")` or `("style", "style.css")`.

	# Parameters:
	- `html : str`
		input HTML content.
	- `assets : list[tuple[Literal["script", "style"], str]]`
		List of (tag_type, filename) tuples to inline.

	# Returns:
	`str` : Modified HTML content with inlined assets.
	"""
	for tag_type, filename in assets:
		if tag_type not in ("style", "script"):
			err_msg: str = f"Unsupported tag type: {tag_type}"
			raise ValueError(err_msg)

		# Dynamically create the pattern for the given tag and filename
		pattern: str = rf'<{tag_type} src="{filename}"></{tag_type}>'
		# assert it's in the text exactly once
		assert html.count(pattern) == 1, (
			f"Pattern {pattern} should be in the html exactly once, found {html.count(pattern) = }"
		)
		# read the content and create the replacement
		content: str = (base_path / filename).read_text()
		replacement: str = f"<{tag_type}>\n{content}\n</{tag_type}>"
		# perform the replacement
		html = html.replace(pattern, replacement)

	return html


def inline_html_file(
	html_path: Path,
	output_path: Path,
) -> None:
	base_path: Path = html_path.parent
	# read the HTML file
	html: str = html_path.read_text()
	# read the assets
	assets: list[tuple[Literal["script", "style"], str]] = []
	for asset in base_path.glob("*.js"):
		assets.append(("script", asset.name))
	for asset in base_path.glob("*.css"):
		assets.append(("style", asset.name))
	# inline the assets
	html_new: str = inline_html_assets(html, assets, base_path)
	# write the new HTML file
	output_path.write_text(html_new)


if __name__ == "__main__":
	import argparse
	
	parser: argparse.ArgumentParser = argparse.ArgumentParser(
		description="Inline local CSS/JS files into an HTML document."
	)
	parser.add_argument(
		"-i",
		"--input-path",
		type=Path,
		help="Path to the HTML file to process.",
	)
	parser.add_argument(
		"-o",
		"--output-path",
		type=str,
		help="Path to save the modified HTML file.",
	)

	args: argparse.Namespace = parser.parse_args()

	inline_html_file(
		html_path=Path(args.input_path),
		output_path=Path(args.output_path),
	)