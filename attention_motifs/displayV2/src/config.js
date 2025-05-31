CONFIG = {
	"dataFile": "pca.jsonl",
	"numericalPrefix": "pc.",
	"defaultColorColumn": "activation.model",
	"defaultSelectionColumn": "activation.model",
	"hoverColumns": [
		"activation.cls",
		"activation.prompt"
	]
};

async function getConfig() {
	try {
		const r = await fetch('config.json');
		if (r.ok) 
		{
			const cfg_load = await r.json();
			// assert fields exist
			for (const key of Object.keys(CONFIG)) {
				if (!(key in cfg_load)) {
					console.warn(`Config key ${key} not found in config.json, using default value`);
					cfg_load[key] = CONFIG[key];
				}
			}
			CONFIG = cfg_load;
			return CONFIG;
		}
		else {
			console.warn('config.json not found, using defaults');
			return CONFIG;
		}
	} catch (e) { console.error('Config load error', e); }
}