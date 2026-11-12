/**
 * SemanticSearch AI — Production JavaScript
 * Handles:
 *  - Semantic search execution via /api/search
 *  - Backend health monitoring via /api/health
 *  - Result card generation with cosine score visualizer
 *  - Real-time client-side category and score threshold filtering
 *  - Document text expand/collapse with query term highlighting
 *  - Reliable clipboard copy with fallback
 *  - Persistent search history in localStorage with per-item removal
 */

"use strict";

(function () {
  /* ============================================================
     1. Constants & Configuration
     ============================================================ */
  const API_SEARCH = "/api/search";
  const API_HEALTH = "/api/health";
  const HISTORY_KEY = "semantic_similarity_search_history";
  const MAX_HISTORY_ITEMS = 10;
  const PREVIEW_CHAR_LIMIT = 320;

  /* ============================================================
     2. Application State
     ============================================================ */
  const state = {
    allResults: [],
    lastQuery: "",
    lastExecMs: 0,
    isSearching: false,
  };

  /* ============================================================
     3. DOM Element Cache
     ============================================================ */
  let dom = {};

  function initDomElements() {
    dom = {
      // Inputs & Buttons
      searchInput: document.getElementById("search-input"),
      clearBtn: document.getElementById("clear-btn"),
      searchBtn: document.getElementById("search-btn"),
      topNSelect: document.getElementById("top-n-select"),
      categoryFilter: document.getElementById("category-filter"),
      categoryWrap: document.getElementById("category-filter-wrap"),
      scoreThreshold: document.getElementById("score-threshold"),
      thresholdLabel: document.getElementById("threshold-label"),
      suggestionsRow: document.getElementById("suggestions-row"),
      inputGroup: document.querySelector(".search-input-group"),

      // Stats Strip
      statsStrip: document.getElementById("stats-strip"),
      statQuery: document.getElementById("stat-query"),
      statCount: document.getElementById("stat-count"),
      statTime: document.getElementById("stat-time"),

      // Results & Alerts
      loadingIndicator: document.getElementById("loading-indicator"),
      errorBanner: document.getElementById("error-banner"),
      errorMessage: document.getElementById("error-message"),
      resultsHeader: document.getElementById("results-header"),
      resultsContainer: document.getElementById("results-container"),
      resultsCountBadge: document.getElementById("results-count-badge"),
      resultsFilterNote: document.getElementById("results-filter-note"),
      noResults: document.getElementById("no-results"),

      // History
      historySection: document.getElementById("history-section"),
      historyList: document.getElementById("history-list"),
      clearHistoryBtn: document.getElementById("clear-history-btn"),

      // Status
      navStatusBadge: document.getElementById("nav-status-badge"),
    };
  }

  /* ============================================================
     4. Health Check
     ============================================================ */
  async function checkBackendHealth() {
    if (!dom.navStatusBadge) return;

    try {
      const response = await fetch(API_HEALTH);
      const data = await response.json();

      if (response.ok && data.model_loaded) {
        setNavStatus("healthy", "Model Ready", "Word2Vec 100-dim model loaded");
      } else {
        setNavStatus("degraded", "Model Degraded", data.error || "Model unavailable");
      }
    } catch {
      setNavStatus("degraded", "Offline", "Cannot connect to server");
    }
  }

  function setNavStatus(statusClass, label, title) {
    if (!dom.navStatusBadge) return;
    dom.navStatusBadge.className = `badge badge-status badge-${statusClass}`;
    const icon =
      statusClass === "healthy"
        ? '<i class="fa-solid fa-circle-check me-1"></i>'
        : '<i class="fa-solid fa-circle-exclamation me-1"></i>';
    dom.navStatusBadge.innerHTML = `${icon}${escapeHtml(label)}`;
    dom.navStatusBadge.title = title || "";
  }

  /* ============================================================
     5. Search Execution
     ============================================================ */
  async function performSearch() {
    const rawQuery = dom.searchInput ? dom.searchInput.value.trim() : "";
    const topN = dom.topNSelect ? parseInt(dom.topNSelect.value, 10) || 5 : 5;

    if (!rawQuery) {
      triggerInputValidation();
      showError("Please enter a search query or click one of the suggestion topics below.");
      return;
    }

    setSearchingState(true);
    hideError();
    hideResults();

    try {
      const response = await fetch(API_SEARCH, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          query: rawQuery,
          top_n: topN,
        }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        const errorMsg =
          data.error || `Search failed with status code ${response.status}.`;
        showError(errorMsg);
        return;
      }

      state.allResults = Array.isArray(data.results) ? data.results : [];
      state.lastQuery = data.query || rawQuery;
      state.lastExecMs = data.execution_time_ms || 0;

      // Update history in localStorage
      addQueryToHistory(state.lastQuery, state.allResults.length);

      // Update statistics strip
      updateStatsBar(data);

      // Populate category filter options based on real results
      populateCategories(state.allResults);

      // Render the result cards
      renderFilteredResults();

      // Scroll smoothly to results so the user immediately sees them
      setTimeout(() => {
        const target =
          (dom.statsStrip && dom.statsStrip.style.display !== "none" ? dom.statsStrip : null) ||
          (dom.resultsHeader && dom.resultsHeader.style.display !== "none" ? dom.resultsHeader : null) ||
          dom.noResults;
        if (target) {
          target.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 60);
    } catch (err) {
      showError(
        "Unable to connect to the backend server. Please verify Flask is running on http://127.0.0.1:5000."
      );
    } finally {
      setSearchingState(false);
    }
  }

  /* ============================================================
     6. Rendering Results & Filters
     ============================================================ */
  function renderFilteredResults() {
    if (!dom.resultsContainer) return;

    const minScoreThreshold = dom.scoreThreshold
      ? parseFloat(dom.scoreThreshold.value) / 100
      : 0;
    const selectedCategory = dom.categoryFilter
      ? dom.categoryFilter.value.trim()
      : "";

    // Filter results according to category and threshold
    const filtered = state.allResults.filter((item) => {
      const passesScore = item.similarity_score >= minScoreThreshold;
      const passesCategory =
        !selectedCategory || item.category === selectedCategory;
      return passesScore && passesCategory;
    });

    dom.resultsContainer.innerHTML = "";

    // Handle empty cases
    if (filtered.length === 0) {
      if (dom.resultsHeader) dom.resultsHeader.style.display = "none";
      if (dom.noResults) {
        dom.noResults.style.display = "block";
        const h5 = dom.noResults.querySelector("h5");
        const p = dom.noResults.querySelector("p");

        if (state.allResults.length === 0) {
          if (h5) h5.textContent = "No matching documents found";
          if (p)
            p.textContent =
              "The query words were not found in the Word2Vec vocabulary. Try rephrasing with different words.";
        } else {
          if (h5) h5.textContent = "No results match your active filters";
          if (p)
            p.textContent =
              "Try lowering the minimum similarity threshold or clearing the category filter.";
        }
      }
      return;
    }

    if (dom.noResults) dom.noResults.style.display = "none";
    if (dom.resultsHeader) dom.resultsHeader.style.display = "flex";

    // Update result counters
    const hiddenCount = state.allResults.length - filtered.length;
    if (dom.resultsCountBadge) {
      dom.resultsCountBadge.textContent = filtered.length;
    }
    if (dom.resultsFilterNote) {
      dom.resultsFilterNote.textContent =
        hiddenCount > 0
          ? `(${hiddenCount} result${hiddenCount > 1 ? "s" : ""} hidden by filters)`
          : "";
    }

    if (dom.statCount) {
      dom.statCount.textContent = `${filtered.length} / ${state.allResults.length}`;
    }

    // Build and append each result card
    filtered.forEach((result, idx) => {
      const card = createResultCardElement(result, idx, state.lastQuery);
      dom.resultsContainer.appendChild(card);
    });
  }

  /**
   * Constructs a secure, interactive DOM element for a search result card.
   * Direct event listeners are attached to avoid quote-escaping issues.
   */
  function createResultCardElement(result, index, query) {
    const card = document.createElement("div");
    card.className = "result-card card p-3 p-md-4";

    const similarity = typeof result.similarity_score === "number" ? result.similarity_score : 0;
    const scorePct = Math.round(similarity * 100);

    // Color theme based on similarity score
    let scoreClass = "score-low";
    let matchQuality = "Low Semantic Match";
    if (scorePct >= 70) {
      scoreClass = "score-high";
      matchQuality = "Strong Semantic Match";
    } else if (scorePct >= 40) {
      scoreClass = "score-medium";
      matchQuality = "Moderate Semantic Match";
    }

    // Rank badge styling
    const rankNum = result.rank || index + 1;
    let rankBadgeClass = "rank-other";
    if (rankNum === 1) rankBadgeClass = "rank-1";
    else if (rankNum === 2) rankBadgeClass = "rank-2";
    else if (rankNum === 3) rankBadgeClass = "rank-3";

    const fullText = (result.text || "").toString();
    const isLongText = fullText.length > PREVIEW_CHAR_LIMIT;
    const previewText = isLongText
      ? fullText.slice(0, PREVIEW_CHAR_LIMIT) + "…"
      : fullText;

    // Highlight query keywords in the snippet
    const highlightedPreviewHtml = highlightQueryTerms(previewText, query);
    const highlightedFullHtml = highlightQueryTerms(fullText, query);

    card.innerHTML = `
      <div class="d-flex flex-wrap gap-3 align-items-start">
        <!-- Rank Badge -->
        <div class="rank-badge ${rankBadgeClass}" title="Rank #${rankNum}">
          #${rankNum}
        </div>

        <!-- Main Body -->
        <div class="flex-grow-1" style="min-width: 0;">
          <div class="d-flex flex-wrap align-items-center gap-2 mb-2">
            <span class="category-pill">
              <i class="fa-solid fa-folder me-1" style="font-size:0.7rem;"></i>
              ${escapeHtml(result.category || "uncategorized")}
            </span>
            <span class="text-muted small">
              Doc ID: <code>${escapeHtml(String(result.document_id))}</code>
            </span>
          </div>

          <!-- Document Text Container -->
          <div class="doc-text-wrapper">
            <p class="doc-text-preview ${isLongText ? "collapsed" : ""}">${highlightedPreviewHtml}</p>
          </div>

          <!-- Card Actions -->
          <div class="d-flex flex-wrap align-items-center gap-2">
            ${
              isLongText
                ? `<button type="button" class="btn-toggle-expand">
                     <i class="fa-solid fa-chevron-down"></i>
                     <span>Show more</span>
                   </button>`
                : ""
            }
            <button type="button" class="btn-doc-action ms-auto btn-copy">
              <i class="fa-regular fa-copy"></i>
              <span>Copy Text</span>
            </button>
          </div>
        </div>

        <!-- Similarity Score Box -->
        <div class="score-box ${scoreClass}">
          <div class="score-header">
            <span class="score-title">Cosine Sim</span>
            <span class="score-percentage">${scorePct}%</span>
          </div>
          <div class="score-progress">
            <div class="score-fill" style="width: 0%" data-target-width="${scorePct}%"></div>
          </div>
          <div class="score-desc">${matchQuality}</div>
        </div>
      </div>
    `;

    // Animate score bar fill smoothly
    requestAnimationFrame(() => {
      const fill = card.querySelector(".score-fill");
      if (fill) {
        requestAnimationFrame(() => {
          fill.style.width = fill.getAttribute("data-target-width");
        });
      }
    });

    // Wire up Expand/Collapse button safely
    const toggleBtn = card.querySelector(".btn-toggle-expand");
    if (toggleBtn) {
      const textElem = card.querySelector(".doc-text-preview");
      toggleBtn.addEventListener("click", () => {
        const isCollapsed = textElem.classList.contains("collapsed");
        if (isCollapsed) {
          textElem.classList.remove("collapsed");
          textElem.innerHTML = highlightedFullHtml;
          toggleBtn.innerHTML =
            '<i class="fa-solid fa-chevron-up"></i><span>Show less</span>';
        } else {
          textElem.classList.add("collapsed");
          textElem.innerHTML = highlightedPreviewHtml;
          toggleBtn.innerHTML =
            '<i class="fa-solid fa-chevron-down"></i><span>Show more</span>';
        }
      });
    }

    // Wire up Copy button safely
    const copyBtn = card.querySelector(".btn-copy");
    if (copyBtn) {
      copyBtn.addEventListener("click", async () => {
        const success = await copyToClipboard(fullText);
        if (success) {
          copyBtn.classList.add("btn-copied");
          copyBtn.innerHTML =
            '<i class="fa-solid fa-check"></i><span>Copied!</span>';
          setTimeout(() => {
            copyBtn.classList.remove("btn-copied");
            copyBtn.innerHTML =
              '<i class="fa-regular fa-copy"></i><span>Copy Text</span>';
          }, 2000);
        } else {
          copyBtn.innerHTML =
            '<i class="fa-solid fa-xmark"></i><span>Failed</span>';
          setTimeout(() => {
            copyBtn.innerHTML =
              '<i class="fa-regular fa-copy"></i><span>Copy Text</span>';
          }, 2000);
        }
      });
    }

    return card;
  }

  /* ============================================================
     7. Keyword Highlighting
     ============================================================ */
  function highlightQueryTerms(text, query) {
    if (!text) return "";
    const escaped = escapeHtml(text);
    if (!query) return escaped;

    // Extract search terms (3+ letters to avoid highlighting single characters)
    const terms = query
      .toLowerCase()
      .split(/\s+/)
      .map((t) => t.replace(/[^a-z0-9]/gi, ""))
      .filter((t) => t.length > 2);

    if (terms.length === 0) return escaped;

    try {
      const pattern = new RegExp(`\\b(${terms.map(escapeRegExp).join("|")})\\b`, "gi");
      return escaped.replace(
        pattern,
        '<mark class="term-highlight">$1</mark>'
      );
    } catch {
      return escaped;
    }
  }

  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  /* ============================================================
     8. Category Filter Population
     ============================================================ */
  function populateCategories(results) {
    if (!dom.categoryFilter || !dom.categoryWrap) return;

    const categories = [
      ...new Set(results.map((r) => r.category).filter(Boolean)),
    ].sort();

    const previousValue = dom.categoryFilter.value;
    dom.categoryFilter.innerHTML = '<option value="">All categories</option>';

    categories.forEach((cat) => {
      const option = document.createElement("option");
      option.value = cat;
      option.textContent = cat;
      if (cat === previousValue) option.selected = true;
      dom.categoryFilter.appendChild(option);
    });

    // Only show category filter if there are 2 or more distinct categories
    dom.categoryWrap.style.setProperty(
      "display",
      categories.length > 1 ? "flex" : "none",
      "important"
    );
  }

  /* ============================================================
     9. Statistics Strip
     ============================================================ */
  function updateStatsBar(data) {
    if (!dom.statsStrip) return;
    dom.statsStrip.style.display = "block";

    if (dom.statQuery) {
      dom.statQuery.textContent = `"${data.query || ""}"`;
      dom.statQuery.title = data.query || "";
    }
    if (dom.statTime) {
      dom.statTime.textContent = `${data.execution_time_ms} ms`;
    }
  }

  /* ============================================================
     10. Search History (localStorage)
     ============================================================ */
  function getSavedHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function persistHistory(items) {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
    } catch (e) {
      console.warn("Could not save search history to localStorage:", e);
    }
  }

  function addQueryToHistory(query, matchCount) {
    if (!query) return;
    let list = getSavedHistory().filter(
      (item) => item.query.toLowerCase() !== query.toLowerCase()
    );

    list.unshift({
      query: query,
      timestamp: new Date().toISOString(),
      count: matchCount || 0,
    });

    list = list.slice(0, MAX_HISTORY_ITEMS);
    persistHistory(list);
    renderSearchHistory();
  }

  function removeHistoryItem(queryToRemove) {
    let list = getSavedHistory().filter(
      (item) => item.query !== queryToRemove
    );
    persistHistory(list);
    renderSearchHistory();
  }

  function clearAllHistory() {
    try {
      localStorage.removeItem(HISTORY_KEY);
    } catch {}
    renderSearchHistory();
  }

  function renderSearchHistory() {
    if (!dom.historySection || !dom.historyList) return;

    const list = getSavedHistory();
    if (list.length === 0) {
      dom.historySection.style.display = "none";
      dom.historyList.innerHTML = "";
      return;
    }

    dom.historySection.style.display = "block";
    dom.historyList.innerHTML = "";

    list.forEach((item) => {
      const li = document.createElement("li");
      li.className = "history-item";

      const timeFormatted = formatRelativeTime(item.timestamp);

      li.innerHTML = `
        <div class="history-item-left">
          <i class="fa-solid fa-clock-rotate-left text-muted"></i>
          <span class="history-query-text">${escapeHtml(item.query)}</span>
        </div>
        <div class="d-flex align-items-center">
          <span class="history-meta me-2">${timeFormatted}</span>
          <button type="button" class="btn-history-remove" title="Remove this item">
            <i class="fa-solid fa-xmark"></i>
          </button>
        </div>
      `;

      // Click to re-search
      li.querySelector(".history-item-left").addEventListener("click", () => {
        if (dom.searchInput) {
          dom.searchInput.value = item.query;
          updateClearButtonVisibility();
          performSearch();
          window.scrollTo({ top: 0, behavior: "smooth" });
        }
      });

      // Click to delete individual item
      li.querySelector(".btn-history-remove").addEventListener("click", (e) => {
        e.stopPropagation();
        removeHistoryItem(item.query);
      });

      dom.historyList.appendChild(li);
    });
  }

  function formatRelativeTime(isoString) {
    if (!isoString) return "";
    const date = new Date(isoString);
    const now = new Date();
    const diffSeconds = Math.floor((now - date) / 1000);

    if (diffSeconds < 60) return "Just now";
    if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
    if (diffSeconds < 86400) return `${Math.floor(diffSeconds / 3600)}h ago`;
    return date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    });
  }

  /* ============================================================
     11. UI State & Utilities
     ============================================================ */
  function setSearchingState(isLoading) {
    state.isSearching = isLoading;

    if (dom.loadingIndicator) {
      dom.loadingIndicator.style.display = isLoading ? "block" : "none";
    }

    if (dom.searchBtn) {
      dom.searchBtn.disabled = isLoading;
      dom.searchBtn.innerHTML = isLoading
        ? '<i class="fa-solid fa-circle-notch fa-spin me-2"></i>Searching…'
        : '<i class="fa-solid fa-magnifying-glass me-2"></i>Search';
    }
  }

  function hideResults() {
    if (dom.resultsContainer) dom.resultsContainer.innerHTML = "";
    if (dom.resultsHeader) dom.resultsHeader.style.display = "none";
    if (dom.noResults) dom.noResults.style.display = "none";
    if (dom.statsStrip) dom.statsStrip.style.display = "none";
  }

  function showError(msg) {
    if (dom.errorMessage) dom.errorMessage.textContent = msg;
    if (dom.errorBanner) dom.errorBanner.style.display = "block";
  }

  function hideError() {
    if (dom.errorBanner) dom.errorBanner.style.display = "none";
    if (dom.errorMessage) dom.errorMessage.textContent = "";
  }

  function triggerInputValidation() {
    if (!dom.searchInput) return;
    dom.searchInput.focus();

    if (dom.inputGroup) {
      dom.inputGroup.classList.add("shake-error");
      setTimeout(() => {
        dom.inputGroup.classList.remove("shake-error");
      }, 600);
    }
  }

  function updateClearButtonVisibility() {
    if (!dom.searchInput || !dom.clearBtn) return;
    dom.clearBtn.style.display =
      dom.searchInput.value.length > 0 ? "inline-flex" : "none";
  }

  async function copyToClipboard(text) {
    // 1. Try standard Navigator API
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch {}
    }

    // 2. Fallback using temporary textarea
    try {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.style.position = "fixed";
      textarea.style.top = "-9999px";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.focus();
      textarea.select();
      const successful = document.execCommand("copy");
      document.body.removeChild(textarea);
      return successful;
    } catch {
      return false;
    }
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /* ============================================================
     12. Event Listeners Initialization
     ============================================================ */
  function setupEventListeners() {
    // Search Button
    if (dom.searchBtn) {
      dom.searchBtn.addEventListener("click", performSearch);
    }

    // Input Keydown & Change
    if (dom.searchInput) {
      dom.searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          performSearch();
        } else if (e.key === "Escape") {
          dom.searchInput.value = "";
          updateClearButtonVisibility();
          hideResults();
        }
      });

      dom.searchInput.addEventListener("input", updateClearButtonVisibility);
    }

    // Clear Button
    if (dom.clearBtn) {
      dom.clearBtn.addEventListener("click", () => {
        if (dom.searchInput) {
          dom.searchInput.value = "";
          dom.searchInput.focus();
        }
        updateClearButtonVisibility();
        hideResults();
        hideError();
      });
    }

    // Score Threshold Slider
    if (dom.scoreThreshold) {
      dom.scoreThreshold.addEventListener("input", () => {
        const value = dom.scoreThreshold.value;
        if (dom.thresholdLabel) dom.thresholdLabel.textContent = `${value}%`;
        if (state.allResults.length > 0) {
          renderFilteredResults();
        }
      });
    }

    // Category Filter Dropdown
    if (dom.categoryFilter) {
      dom.categoryFilter.addEventListener("change", () => {
        if (state.allResults.length > 0) {
          renderFilteredResults();
        }
      });
    }

    // Result count selector (top_n) change
    if (dom.topNSelect) {
      dom.topNSelect.addEventListener("change", () => {
        if (dom.searchInput && dom.searchInput.value.trim().length > 0) {
          performSearch();
        }
      });
    }

    // Suggestion Chips
    document.querySelectorAll(".suggestion-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const queryText = chip.getAttribute("data-query");
        if (dom.searchInput && queryText) {
          dom.searchInput.value = queryText;
          updateClearButtonVisibility();
          performSearch();
        }
      });
    });

    // Clear History Button
    if (dom.clearHistoryBtn) {
      dom.clearHistoryBtn.addEventListener("click", clearAllHistory);
    }
  }

  /* ============================================================
     13. App Bootstrapping
     ============================================================ */
  document.addEventListener("DOMContentLoaded", () => {
    initDomElements();
    setupEventListeners();
    updateClearButtonVisibility();
    renderSearchHistory();
    checkBackendHealth();
  });
})();
