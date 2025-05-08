function computeCombinedUrl(baseUrl, selectedValues, selectionColumn) {
	if (!selectedValues?.length) return baseUrl;

	const baseUrlObj = new URL(baseUrl, window.location.href);
	const params = baseUrlObj.searchParams;

	if (selectionColumn === 'activation.model' || selectionColumn === 'activation.cls') {
		const modelHeadMap = {};
		selectedValues.forEach(v => {
			const [model, l, h] = v.split(':');
			if (!model || !l || !h) return;
			(modelHeadMap[model] ??= []).push(`${l}${h}`);
		});
		const models = Object.keys(modelHeadMap);
		if (models.length) {
			params.set('models', models.join('~'));
			models.forEach(m => params.set(`heads-${m}`, modelHeadMap[m].join('~')));
		}
	} else if (selectionColumn === 'activation.prompt') {
		params.set('prompts', selectedValues.join('~'));
	} else {
		params.set(selectionColumn.split('.').pop(), selectedValues.join('~'));
	}

	return baseUrlObj.toString();
}