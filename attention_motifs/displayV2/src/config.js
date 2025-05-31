/* global, mutable CONFIG + helper to merge an optional config.json */
export let CONFIG = {
	dataFile: "pca.jsonl",
	numericalPrefix: "pc.",
	defaultColorColumn: "activation.model",
	defaultSelectionColumn: "activation.model",
	hoverColumns: ["activation.cls", "activation.prompt"]
};

/**
 * Load config.json (if present) and merge into CONFIG.
 * Missing keys fall back to the defaults above.
 * @returns {Promise<object>} resolved CONFIG object
 */
export async function getConfig() {
	try {
		const r = await fetch("config.json");
		if (!r.ok) {
			console.warn("config.json not found, using defaults");
			return CONFIG;
		}

		const loaded = await r.json();

		/* copy defaults for any missing keys */
		for (const k of Object.keys(CONFIG)) {
			if (!(k in loaded)) {
				console.warn(`Config key '${k}' missing – using default`);
				loaded[k] = CONFIG[k];
			}
		}

		/* mutate, don’t replace – so live bindings stay valid */
		Object.assign(CONFIG, loaded);
	} catch (e) {
		console.error("Config load error:", e);
	}

	return CONFIG;
}
