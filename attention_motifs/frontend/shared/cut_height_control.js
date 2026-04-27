/**
 * Shared cut-height slider + number input control.
 *
 * Used by both clustering and cluster_trends views to avoid duplicating
 * the bidirectional slider/input sync logic.
 */

/**
 * Create a cut-height control with a range slider and number input.
 * @param {Object} options
 * @param {HTMLElement} options.container - DOM element to render into
 * @param {number} [options.maxHeight=10] - Maximum height value
 * @param {number} [options.step=0.01] - Step size for the slider
 * @param {number} [options.initialValue=5] - Initial cut height
 * @param {string} [options.label="Cut Height:"] - Label text
 * @param {boolean} [options.showTicks=false] - Whether to include a ticks overlay container
 * @param {(height: number) => void} options.onChange - Callback when value changes
 * @returns {{
 *   setValue: (h: number) => void,
 *   getValue: () => number,
 *   setDisabled: (b: boolean) => void,
 *   getSliderElement: () => HTMLInputElement,
 *   getTicksContainer: () => HTMLElement|null,
 * }}
 */
function createCutHeightControl(options) {
  const {
    container,
    maxHeight = 10,
    step = 0.01,
    initialValue = 5,
    label = "Cut Height:",
    showTicks = false,
    onChange,
  } = options;

  // Create label
  const labelEl = document.createElement("label");
  labelEl.setAttribute("for", "cut-height");
  labelEl.textContent = label;
  container.appendChild(labelEl);

  // Create slider wrapper (with optional ticks overlay)
  let ticksContainer = null;
  const sliderWrapper = document.createElement("div");
  sliderWrapper.className = "slider-with-ticks";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.id = "cut-height";
  slider.min = "0";
  slider.max = String(maxHeight);
  slider.step = String(step);
  slider.value = String(initialValue);
  sliderWrapper.appendChild(slider);

  if (showTicks) {
    ticksContainer = document.createElement("div");
    ticksContainer.className = "slider-ticks";
    ticksContainer.id = "cut-height-ticks";
    sliderWrapper.appendChild(ticksContainer);
  }

  container.appendChild(sliderWrapper);

  // Create number input
  const numInput = document.createElement("input");
  numInput.type = "number";
  numInput.id = "cut-height-input";
  numInput.min = "0";
  numInput.max = String(maxHeight);
  numInput.step = String(step);
  numInput.value = initialValue.toFixed(3);
  numInput.className = "control-input";
  numInput.style.width = "60px";
  container.appendChild(numInput);

  // Bidirectional sync
  slider.addEventListener("input", (e) => {
    const h = parseFloat(e.target.value);
    numInput.value = h.toFixed(3);
    if (onChange) onChange(h);
  });

  numInput.addEventListener("change", (e) => {
    const h = Math.max(0, Math.min(maxHeight, parseFloat(e.target.value) || 0));
    numInput.value = h.toFixed(3);
    slider.value = String(h);
    if (onChange) onChange(h);
  });

  return {
    /** @param {number} h */
    setValue(h) {
      slider.value = String(h);
      numInput.value = h.toFixed(3);
    },
    /** @returns {number} */
    getValue() {
      return parseFloat(slider.value);
    },
    /** @param {boolean} b */
    setDisabled(b) {
      slider.disabled = b;
      numInput.disabled = b;
    },
    /** @returns {HTMLInputElement} */
    getSliderElement() {
      return slider;
    },
    /** @returns {HTMLElement|null} */
    getTicksContainer() {
      return ticksContainer;
    },
  };
}
