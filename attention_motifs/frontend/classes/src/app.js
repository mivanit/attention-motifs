let classificationsData = null;
let paperFilters = new Set();

async function loadClassifications() {
  try {
    const response = await fetch("../vis/attnpedia/ap.json");
    if (!response.ok) {
      throw new Error("Failed to load classification data");
    }
    classificationsData = await response.json();

    // Initialize paper filters
    initializePaperFilters();

    // Display classifications
    displayClassifications();

    // Set up search functionality
    setupSearch();
  } catch (error) {
    console.error("Error loading classifications:", error);
    document.getElementById("classifications-container").innerHTML =
      '<div class="error">Error loading classification data. Please check that ap.json exists.</div>';
  }
}

function initializePaperFilters() {
  const papers = new Set();

  // Extract unique paper prefixes from classification types
  Object.keys(classificationsData.type_metadata).forEach((type) => {
    const paperPrefix = type.split(":")[0];
    papers.add(paperPrefix);
  });

  // Initialize all papers as selected
  paperFilters = new Set(papers);

  // Create checkboxes
  const checkboxContainer = document.getElementById("paper-checkboxes");
  let html = "";

  papers.forEach((paper) => {
    html += `
            <span class="paper-checkbox">
                <input type="checkbox" 
                       id="paper-${paper}" 
                       value="${paper}" 
                       checked 
                       onchange="togglePaper('${paper}')">
                <label for="paper-${paper}">${paper}</label>
            </span>
        `;
  });

  checkboxContainer.innerHTML = html;
}

function togglePaper(paper) {
  if (paperFilters.has(paper)) {
    paperFilters.delete(paper);
  } else {
    paperFilters.add(paper);
  }
  displayClassifications();
}

function setupSearch() {
  const searchInput = document.getElementById("search-input");
  searchInput.addEventListener("input", (e) => {
    displayClassifications(e.target.value.toLowerCase());
  });
}

function displayClassifications(searchTerm = "") {
  const container = document.getElementById("classifications-container");
  const typeMetadata = classificationsData.type_metadata;

  // Group classifications by paper
  const paperGroups = {};

  Object.entries(typeMetadata).forEach(([type, meta]) => {
    const [paper, classification] = type.split(":");

    // Apply filters
    if (!paperFilters.has(paper)) return;
    if (searchTerm && !type.toLowerCase().includes(searchTerm)) return;

    if (!paperGroups[paper]) {
      paperGroups[paper] = {
        url: meta.url,
        notes: meta.notes,
        model: meta.model,
        classifications: [],
      };
    }

    paperGroups[paper].classifications.push({
      type: type,
      name: classification,
      heads: meta.heads,
      n_heads: meta.n_heads,
    });
  });

  // Update stats
  updateStats(paperGroups);

  // Generate HTML
  if (Object.keys(paperGroups).length === 0) {
    container.innerHTML =
      '<div class="no-results">No classifications match your search criteria.</div>';
    return;
  }

  let html = "";

  Object.entries(paperGroups).forEach(([paper, data]) => {
    html += `
            <div class="paper-section">
                <div class="paper-title">
                    ${paper}
                    <span class="head-count">${data.classifications.length} classifications</span>
                </div>
                <div class="paper-info">
                    ${data.notes ? `<div>Note: ${data.notes}</div>` : ""}
                    <div>Model: ${data.model}</div>
                    <div>Paper: ${data.url}</div>
                </div>
                <div class="classification-grid">
        `;

    data.classifications.forEach((classification) => {
      const attnPediaUrl = `../vis/attnpedia/index.html?classification_mode=true&current_classification=${encodeURIComponent(classification.type)}`;

      html += `
                <a href="${attnPediaUrl}" class="classification-item">
                    <div class="classification-name">
                        ${classification.name}
                        <span class="head-count">${classification.n_heads}</span>
                    </div>
                    <div class="classification-heads">
                        ${classification.heads
                          .slice(0, 3)
                          .map((h) => h.split(":").slice(1).join(":"))
                          .join(", ")}
                        ${classification.heads.length > 3 ? "..." : ""}
                    </div>
                </a>
            `;
    });

    html += `
                </div>
            </div>
        `;
  });

  container.innerHTML = html;
}

function updateStats(paperGroups) {
  const totalPapers = Object.keys(paperGroups).length;
  const totalClassifications = Object.values(paperGroups).reduce(
    (sum, paper) => sum + paper.classifications.length,
    0,
  );
  const totalHeads = Object.values(paperGroups).reduce(
    (sum, paper) =>
      sum + paper.classifications.reduce((s, c) => s + c.n_heads, 0),
    0,
  );

  // Get unique models
  const models = new Set();
  Object.values(paperGroups).forEach((paper) => {
    models.add(paper.model);
  });

  document.getElementById("stats").innerHTML = `
        Showing ${totalClassifications} classifications from ${totalPapers} papers, 
        covering ${totalHeads} attention heads across ${models.size} models
    `;
}

// Load data when page loads
document.addEventListener("DOMContentLoaded", loadClassifications);
