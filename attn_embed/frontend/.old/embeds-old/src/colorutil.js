/**
 * Converts HSL color values to hexadecimal color representation
 * @param {number} h - Hue value (0-360)
 * @param {number} s - Saturation value (0-100)
 * @param {number} l - Lightness value (0-100)
 * @returns {string} Hexadecimal color string (e.g., "#ff0000")
 */
function hslToHex(h, s, l) {
  s /= 100;
  l /= 100;

  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;

  let r, g, b;

  if (0 <= h && h < 60) {
    [r, g, b] = [c, x, 0];
  } else if (60 <= h && h < 120) {
    [r, g, b] = [x, c, 0];
  } else if (120 <= h && h < 180) {
    [r, g, b] = [0, c, x];
  } else if (180 <= h && h < 240) {
    [r, g, b] = [0, x, c];
  } else if (240 <= h && h < 300) {
    [r, g, b] = [x, 0, c];
  } else {
    [r, g, b] = [c, 0, x];
  }

  const toHex = (value) => {
    const hex = Math.round((value + m) * 255).toString(16);
    return hex.length === 1 ? "0" + hex : hex;
  };

  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

/**
 * Generates an array of visually distinct colors
 * @param {number} count - Number of colors to generate
 * @returns {string[]} Array of hexadecimal color strings
 */
function generateDistinctColors(count) {
  const colors = [];

  for (let i = 0; i < count; i++) {
    // Generate colors with good saturation and brightness
    const h = Math.floor(Math.random() * 360); // Hue (0-360)
    const s = Math.floor(50 + Math.random() * 50); // Saturation (50-100%)
    const l = Math.floor(40 + Math.random() * 20); // Lightness (40-60%)

    // Convert HSL to HEX
    const color = hslToHex(h, s, l);
    colors.push(color);
  }

  return colors;
}
