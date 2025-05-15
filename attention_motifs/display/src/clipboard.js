function formatHoverDataForClipboard(
	hoverData,
	selectionColumn,
	hoverColumns,
) {
	if (!hoverData?.points?.length) return 'No point data available';
	const point = hoverData.points[0];
	const customData = point.customdata ?? [];

	let text = 'Point Info:\n';

	// Include each hover column in order
	hoverColumns.forEach((colName, i) => {
		const val = (customData[i] !== undefined) ? customData[i] : 'N/A';
		text += `${colName}: ${val}\n`;
	});

	// Coordinates
	text += `coord: [${point.x.toFixed(2)}, ${point.y.toFixed(2)}, ${point.z.toFixed(2)}]\n`;

	// If there's one more element in customData beyond hover columns, treat it as selection value
	if (customData.length > hoverColumns.length) {
		const selVal = customData[hoverColumns.length];
		text += `${selectionColumn}: ${selVal}\n`;
	}

	return text;
}

function copyHoverDataToClipboard(
	hoverData,
	selectionColumn,
	hoverColumns,
	cb = () => { },
) {
	if (!hoverData) return;
	try {
		const txt = formatHoverDataForClipboard(hoverData, selectionColumn, hoverColumns);
		navigator.clipboard.writeText(txt).then(
			() => cb('Hover data copied to clipboard'),
			err => { console.error(err); cb('Failed to copy hover data'); }
		);
	} catch (err) {
		console.error(err);
		cb('Error preparing hover data for clipboard');
	}
}