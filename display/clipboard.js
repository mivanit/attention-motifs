function formatHoverDataForClipboard(hoverData, selectionColumn) {
	// Extract the hover text from data
	if (!hoverData?.points?.length) return 'No point data available';
	const point = hoverData.points[0];
	const customData = point.customdata ?? [];
	// Format the text in a clean way
	let text = 'Point Info:\n';
	if (customData.length > 0) text += `head: ${customData[0]}\n`;
	if (customData.length > 1) text += `prompt: ${customData[1]}\n`;
	// Add coordinates
	text += `coord: [${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)}]\n`;
	if (customData.length > 2) text += `${selectionColumn}: ${customData[2]}\n`;
	return text;
}

function copyHoverDataToClipboard(
	hoverData,
	selectionColumn,
	cb = () => { },
) {
	if (!hoverData) return;
	try {
		const txt = formatHoverDataForClipboard(hoverData, selectionColumn);
		navigator.clipboard.writeText(txt).then(
			() => cb('Hover data copied to clipboard'),
			err => { console.error(err); cb('Failed to copy hover data'); }
		);
	} catch (err) {
		console.error(err);
		cb('Error preparing hover data for clipboard');
	}
}