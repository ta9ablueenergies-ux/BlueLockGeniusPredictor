const config = {
    // TOP 5 ELITE LEAGUES
    leagues: [
        'PremierLeague', 'LaLiga', 'SerieA', 'Bundesliga', 'Ligue1',
        // EUROPEAN CUPS
        'ChampionsLeague',
        // SECOND TIER
        'Championship', 'ScottishPremiership', 'Eredivisie', 'LigaNOS', 'BelgianProLeague', 'SuperLig',
        // 2ND DIVISIONS
        '2Bundesliga', 'Ligue2', 'LaLiga2', 'SerieB'
    ],
    defaultLeague: 'PremierLeague',
    apiBase: 'http://localhost:8000'
};

let currentLeague = config.defaultLeague;
let isHistoryMode = false;
window.filterMode = 'exact'; // 'exact' or 'week'

document.addEventListener('DOMContentLoaded', () => {
    init();
    setupPipelineControls();
});

async function init() {
    setupEventListeners();
    const grid = document.getElementById('match-grid');
    grid.innerHTML = `<div class="loading-state"><div class="spinner"></div><p>Synchronizing V5.9 Intelligence Layer...</p></div>`;

    // await triggerAutoRun(); // Removed: User now controls the sweep manually
    
    // Default to Global view
    const globalBtn = document.getElementById('nav-global');
    if (globalBtn) {
        document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
        globalBtn.classList.add('active');
        currentLeague = 'global';
        loadGlobalIntelligence();
        loadTickets('global');
    } else {
        loadLeague(config.defaultLeague);
        loadTickets(config.defaultLeague);
    }
    
    updateLastSyncTime();
}

function setupPipelineControls() {
    const launchButton = document.getElementById('launch-pipeline');
    const purgeButton = document.getElementById('purge-history');
    const statusDiv    = document.getElementById('pipeline-status');
    const messageDiv   = document.getElementById('pipeline-message');
    let pollTimer = null;

    // Pre-fill date inputs with today's date/time range (00:00 → 23:59)
    const now = new Date();
    const pad = n => String(n).padStart(2, '0');
    const todayStart = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}T00:00`;
    const todayEnd   = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}T23:59`;
    const startInput = document.getElementById('start-date');
    const endInput   = document.getElementById('end-date');
    const displayFilterInput = document.getElementById('display-date-filter');
    
    if (startInput && !startInput.value) startInput.value = todayStart;
    if (endInput   && !endInput.value)   endInput.value   = todayEnd;
    if (displayFilterInput && !displayFilterInput.value) displayFilterInput.value = todayStart.split('T')[0];

    if (displayFilterInput) {
        const applyBtn = document.getElementById('apply-display-filter');
        const applyWeekBtn = document.getElementById('apply-this-week-filter');
        const triggerFilter = () => {
            if (isHistoryMode) loadHistory();
            else if (currentLeague === 'global') loadGlobalIntelligence();
            else if (currentLeague === 'v11') loadV11Optimization();
            else if (currentLeague === 'audit') loadShadowAudit();
            else if (currentLeague === 'markets') loadMarketModels();
            else loadLeague(currentLeague);
        };
        
        displayFilterInput.addEventListener('change', () => { window.filterMode = 'exact'; triggerFilter(); });
        if (applyBtn) applyBtn.addEventListener('click', () => { window.filterMode = 'exact'; triggerFilter(); });
        if (applyWeekBtn) applyWeekBtn.addEventListener('click', () => { window.filterMode = 'week'; triggerFilter(); });
        const leagueFilterEl = document.getElementById('global-league-filter');
        if (leagueFilterEl) leagueFilterEl.addEventListener('change', () => { window.filterMode = window.filterMode || 'exact'; triggerFilter(); });
    }

    function stopPolling() {
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    }

    function refreshCurrentView() {
        if (isHistoryMode || currentLeague === 'history') loadHistory();
        else if (currentLeague === 'global') loadGlobalIntelligence();
        else if (currentLeague === 'v11') loadV11Optimization();
        else if (currentLeague === 'audit') loadShadowAudit();
        else if (currentLeague === 'markets') loadMarketModels();
        else loadLeague(currentLeague);
        loadTickets(['global', 'audit', 'markets', 'v11', 'history'].includes(currentLeague) ? 'global' : currentLeague);
    }

    function startPolling() {
        let elapsed = 0;
        pollTimer = setInterval(async () => {
            elapsed += 2;
            try {
                const res  = await fetch('/api/status');
                const body = await res.json();
                if (!body.is_running) {
                    stopPolling();
                    messageDiv.innerHTML = '✅ <strong>Pipeline complete!</strong> Refreshing data...';
                    launchButton.disabled = false;
                    launchButton.textContent = '🚀 Launch Pipeline';
                    // Auto-refresh dashboard
                    setTimeout(() => {
                        refreshCurrentView();
                        updateLastSyncTime();
                        messageDiv.textContent = `✅ Data updated successfully (took ~${elapsed}s).`;
                    }, 1500);
                } else {
                    messageDiv.textContent = `⚙️ Pipeline running… ${elapsed}s elapsed`;
                }
            } catch(e) {
                // Server might still be busy — keep polling
            }
        }, 2000);
    }

    if (launchButton) {
        launchButton.addEventListener('click', async () => {
            const startDate = startInput.value;
            const endDate   = endInput.value;
            const sweepMode = document.getElementById('sweep-mode')?.value || 'hybrid';
            const allowMockFallback = Boolean(document.getElementById('allow-mock-fallback')?.checked);
            const enableBrowserScraping = Boolean(document.getElementById('enable-browser-scraping')?.checked);

            stopPolling();
            statusDiv.style.display = 'block';
            messageDiv.textContent = '🚀 Contacting Intelligence Server...';
            launchButton.disabled = true;
            launchButton.textContent = '🔄 Launching...';

            try {
                const response = await fetch('/api/launch-pipeline', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ startDate, endDate, sweepMode, allowMockFallback, enableBrowserScraping })
                });

                if (response.ok) {
                    const body = await response.json();
                    messageDiv.textContent = `⚙️ ${body.message} — tracking progress...`;
                    // Poll /api/status every 2 s until done
                    startPolling();
                } else if (response.status === 409) {
                    messageDiv.textContent = '⏳ Pipeline already running — polling for completion...';
                    startPolling();
                } else {
                    const errText = await response.text();
                    messageDiv.textContent = '❌ Error: ' + errText;
                    launchButton.disabled = false;
                    launchButton.textContent = '🚀 Launch Pipeline';
                }
            } catch (error) {
                messageDiv.textContent = '❌ Network error — is the API server running? ' + error.message;
                launchButton.disabled = false;
                launchButton.textContent = '🚀 Launch Pipeline';
            }
        });
    }

    if (purgeButton) {
        purgeButton.addEventListener('click', async () => {
            if (confirm('⚠️ Are you sure you want to purge all history? This will delete all existing data.')) {
                try {
                    const response = await fetch('/api/purge-history', {
                        method: 'POST'
                    });

                    if (response.ok) {
                        messageDiv.textContent = '🗑️ History purged successfully. Reloading data...';
                        // Reload the current view
                        setTimeout(() => {
                            refreshCurrentView();
                            messageDiv.textContent = '✅ History purged and data reloaded.';
                        }, 1000);
                    } else {
                        messageDiv.textContent = '❌ Error purging history.';
                    }
                } catch (error) {
                    messageDiv.textContent = '❌ Error: ' + error.message;
                }
            }
        });
    }
}

async function triggerAutoRun() {
    try {
        await fetch('/api/refresh');
    } catch (e) { console.error("Auto-run failed", e); }
}

async function loadTickets(league = 'global') {
    const container = document.getElementById('ticket-carousel');
    const title = document.getElementById('ticket-title');
    const ticketFile = league === 'global' ? 'data/tickets.json' : `data/tickets_${league}.json`;
    
    title.innerText = league === 'global' ? '🚀 Overall Quantum Ticket' : `🚀 ${league.replace(/([A-Z])/g, ' $1').trim()} Specialist`;

    try {
        title.innerText = league === 'global'
            ? 'Global Quantum Ticket'
            : `${league.replace(/([A-Z])/g, ' $1').trim()} Quantum Ticket`;
        const response = await fetch(`${ticketFile}?t=${Date.now()}`);
        if (!response.ok) throw new Error("No specific ticket");
        const tickets = await response.json();
        renderTickets(tickets);
    } catch (e) {
        // Fallback to global if league specific fails
        if (league !== 'global') {
            loadTickets('global');
        } else {
            container.innerHTML = '<div class="ticket-placeholder">Generating Quantum Tickets for this cycle...</div>';
        }
    }
}

function renderTickets(tickets) {
    const container = document.getElementById('ticket-carousel');
    container.innerHTML = '';

    const types = [
        { key: 'TYPE_A_ULTRA_SAFE', label: 'Ultra-Safe', class: 'ultra' },
        { key: 'TYPE_B_BALANCED', label: 'Balanced', class: 'balanced' },
        { key: 'TYPE_C_VALUE', label: 'Value', class: 'value' }
    ];

    const displayFilter = document.getElementById('display-date-filter');
    const targetDate = displayFilter ? displayFilter.value : new Date().toISOString().split('T')[0];

    const seenMatches = new Set();

    types.forEach(type => {
        let ticketMatches = tickets[type.key];
        if (!ticketMatches) return;

        // Filter by Date and De-duplicate
        ticketMatches = ticketMatches.filter(m => {
            const mId = `${m.Home}-${m.Away}-${m.Date}`;
            if (seenMatches.has(mId)) return false;

            let mDate = m.Date || m.date || m.match_date || '';
            if (!mDate) return true;
            
            if (mDate.includes('/')) {
                const parts = mDate.split('/');
                if (parts.length === 3) mDate = `${parts[2]}-${parts[1].padStart(2, '0')}-${parts[0].padStart(2, '0')}`;
            }
            if (mDate.includes('T')) mDate = mDate.split('T')[0];
            
            const tDate = targetDate.split('T')[0];
            const isMatch = mDate === tDate;
            
            if (isMatch) seenMatches.add(mId);
            return isMatch;
        });

        if (ticketMatches.length === 0) return;

        const card = document.createElement('div');
        card.className = `ticket-card ${type.class}`;
        card.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                <span class="ticket-type" style="color:var(--neon-${type.class === 'ultra' ? 'gold' : (type.class === 'balanced' ? 'blue' : 'violet')}); font-weight:700; letter-spacing:1px;">
                    ${type.label}
                </span>
                <span style="font-size:0.6rem; color:var(--text-dim); opacity:0.7">TRUST TICKET</span>
            </div>
            <div class="ticket-legs">
                ${ticketMatches.map(m => `
                    <div class="leg" style="margin-bottom:0.75rem; padding-bottom:0.75rem; border-bottom:1px solid rgba(255,255,255,0.05)">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div class="leg-teams" style="font-weight:600; color:var(--text-bright)">${m.Home} v ${m.Away}</div>
                            <div style="font-size:0.55rem; color:var(--neon-gold); background:rgba(212,175,55,0.1); padding:1px 4px; border-radius:3px">${(m.League || 'GLOBAL').replace(/([A-Z])/g, ' $1').trim()}</div>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:0.25rem;">
                            <div class="leg-market" style="font-size:0.8rem; color:var(--neon-blue)">${m.prediction || m.best_market}</div>
                            <div style="font-size:0.75rem; color:var(--text-dim)">Trust ${trustScore(m)}</div>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:0.2rem; font-size:0.65rem; color:var(--text-dim);">
                            <span>${m.best_market || 'Best Market'}</span>
                            <span>${m.market_probability ? safePct(m.market_probability) : ''}</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        container.appendChild(card);
    });

    if (container.innerHTML === '') {
        container.innerHTML = '<div class="ticket-placeholder">No high-conviction tickets detected in this segment.</div>';
    }
}

function setupEventListeners() {
    document.querySelectorAll('.nav-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const league = e.target.getAttribute('data-league');
            const view = e.target.getAttribute('data-view');
            
            document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');

            if (view === 'history') {
                isHistoryMode = true;
                currentLeague = 'history';
                loadHistory();
            } else if (view === 'global') {
                isHistoryMode = false;
                currentLeague = 'global';
                loadGlobalIntelligence();
                loadTickets('global');
            } else if (view === 'v11') {
                isHistoryMode = false;
                currentLeague = 'v11';
                loadV11Optimization();
            } else if (view === 'audit') {
                isHistoryMode = false;
                currentLeague = 'audit';
                loadShadowAudit();
                loadTickets('global');
            } else if (view === 'markets') {
                isHistoryMode = false;
                currentLeague = 'markets';
                loadMarketModels();
                loadTickets('global');
            } else {
                isHistoryMode = false;
                currentLeague = league;
                loadLeague(league);
                loadTickets(league);
            }
        });
    });
}

async function loadGlobalIntelligence() {
    const grid = document.getElementById('match-grid');
    const title = document.getElementById('league-title');
    const displayFilter = document.getElementById('display-date-filter');
    const targetDate = displayFilter ? displayFilter.value : new Date().toISOString().split('T')[0];

    title.innerText = "Global Market Intelligence";
    grid.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Synchronizing Global Manifest...</p></div>';

    try {
        // Try Live API first
        const apiRes = await fetch(`${config.apiBase}/api/v5/predictions/global?t=${Date.now()}`).catch(() => null);
        let matches = [];
        
        if (apiRes && apiRes.ok) {
            const apiData = await apiRes.json();
            matches = apiData.data;
        } else {
            // Fallback to static manifest
            const res = await fetch(`data/global_manifest.json?t=${Date.now()}`);
            if (!res.ok) throw new Error("Manifest missing");
            const manifest = await res.json();
            matches = manifest.matches;
        }

        // Filter by current target date and optional league
        const leagueFilterEl = document.getElementById('global-league-filter');
        const leagueFilter = leagueFilterEl ? leagueFilterEl.value : '';
        const filtered = matches.filter(m => {
            let mDate = m.Date || m.date || m.match_date || '';
            if (!mDate) return false;

            if (mDate.includes('/')) {
                const parts = mDate.split('/');
                if (parts.length === 3) mDate = `${parts[2]}-${parts[1].padStart(2, '0')}-${parts[0].padStart(2, '0')}`;
            }
            if (mDate.includes('T')) mDate = mDate.split('T')[0];

            const tDate = targetDate.split('T')[0];
            let passDate;
            if (window.filterMode === 'week') {
                const d1 = new Date(tDate + 'T00:00:00');
                const d2 = new Date(mDate + 'T00:00:00');
                const diffDays = Math.floor((d2 - d1) / (1000 * 60 * 60 * 24));
                passDate = diffDays >= 0 && diffDays <= 7;
            } else {
                passDate = mDate === tDate;
            }
            if (!passDate) return false;
            if (leagueFilter && (m.league || m.League || '') !== leagueFilter) return false;
            return true;
        });

        renderMatches(filtered);
        updateHeaderStats(filtered);
    } catch (e) {
        grid.innerHTML = '<div class="loading-state"><p>Global Intelligence not found. Please ensure the API is running or run a full sweep.</p></div>';
    }
}

async function loadShadowAudit() {
    const grid = document.getElementById('match-grid');
    const title = document.getElementById('league-title');
    const ticketGrid = document.getElementById('ticket-carousel');
    
    title.innerText = "Shadow Performance Audit";
    ticketGrid.innerHTML = '<div class="ticket-placeholder">Audit mode active: Intelligence tracking is live.</div>';
    grid.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Querying Shadow Log...</p></div>';

    try {
        const res = await fetch(`data/shadow_log.json?t=${Date.now()}`);
        if (!res.ok) throw new Error("No audit log");
        
        const log = await res.json();
        grid.innerHTML = `
            <div class="audit-container" style="grid-column: 1 / -1; background: rgba(255,255,255,0.02); border-radius: 12px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.05); overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-family: 'Inter';">
                    <thead>
                        <tr style="text-align: left; color: var(--neon-gold); border-bottom: 2px solid rgba(255,255,255,0.1);">
                            <th style="padding: 1rem;">Timestamp</th>
                            <th style="padding: 1rem;">Match</th>
                            <th style="padding: 1rem;">League</th>
                            <th style="padding: 1rem;">Prediction</th>
                            <th style="padding: 1rem;">Prob</th>
                            <th style="padding: 1rem;">Odds</th>
                            <th style="padding: 1rem;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${log.map(entry => `
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); color: var(--text-bright);">
                                <td style="padding: 1rem; font-family: 'Roboto Mono'; font-size: 0.8rem; color: var(--text-dim);">${entry.timestamp}</td>
                                <td style="padding: 1rem; font-weight: 600;">${entry.match}</td>
                                <td style="padding: 1rem;"><span style="background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px; font-size: 0.7rem;">${entry.league}</span></td>
                                <td style="padding: 1rem; color: var(--neon-blue);">${entry.prediction}</td>
                                <td style="padding: 1rem;">${(entry.prob * 100).toFixed(1)}%</td>
                                <td style="padding: 1rem;">${entry.odds}</td>
                                <td style="padding: 1rem;"><span class="badge ${entry.result === 'PENDING' ? 'trap' : 'anchor'}" style="font-size: 0.6rem;">${entry.result}</span></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    } catch (e) {
        grid.innerHTML = '<div class="loading-state"><p>Audit Log not found. Run the pipeline to start tracking performance.</p></div>';
    }
}

async function loadMarketModels() {
    const grid = document.getElementById('match-grid');
    const title = document.getElementById('league-title');
    const ticketGrid = document.getElementById('ticket-carousel');

    title.innerText = "Goals, Corners and Cards Models";
    ticketGrid.innerHTML = '<div class="ticket-placeholder">Count-market layer active: goals, corners and cards are evaluated separately from 1X2.</div>';
    grid.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Loading market model reports...</p></div>';

    try {
        const [marketRes, profileRes, qualityRes] = await Promise.all([
            fetch(`data/market_count_model_report.json?t=${Date.now()}`),
            fetch(`data/league_market_profiles.json?t=${Date.now()}`),
            fetch(`data/probability_quality_report.json?t=${Date.now()}`)
        ]);
        if (!marketRes.ok) throw new Error("No market-count report found");
        const report = await marketRes.json();
        const profiles = profileRes.ok ? await profileRes.json() : {profiles: {}};
        const quality = qualityRes.ok ? await qualityRes.json() : null;
        const rows = Object.values(report.leagues || {}).filter(r => r.status === 'trained');
        const q = quality?.overall?.model || {};

        grid.innerHTML = `
            <div class="audit-container" style="grid-column:1 / -1; background:rgba(255,255,255,0.02); border-radius:12px; padding:1.5rem; border:1px solid rgba(255,255,255,0.06); overflow-x:auto;">
                <div style="display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; margin-bottom:1.5rem;">
                    <div>
                        <h2 style="margin:0; font-size:1.35rem;">Market Count Layer</h2>
                        <p style="margin:0.4rem 0 0; color:var(--text-dim); font-size:0.85rem;">Negative-binomial/Poisson style probabilities for totals, independent from the main result classifier.</p>
                    </div>
                    <span class="engine-badge" style="background:rgba(0,210,255,0.08); color:var(--neon-blue);">${report.created_at || 'latest'}</span>
                </div>
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:1rem; margin-bottom:1.5rem;">
                    <div class="stat-card"><span class="label">Trained Leagues</span><span class="value">${rows.length}</span></div>
                    <div class="stat-card"><span class="label">1X2 Log Loss</span><span class="value">${q.log_loss ?? 'N/A'}</span></div>
                    <div class="stat-card"><span class="label">1X2 Brier</span><span class="value">${q.brier ?? 'N/A'}</span></div>
                    <div class="stat-card"><span class="label">Draw Recall</span><span class="value">${q.draw_recall !== undefined ? (q.draw_recall * 100).toFixed(1) + '%' : 'N/A'}</span></div>
                </div>
                <table style="width:100%; border-collapse:collapse;">
                    <thead>
                        <tr style="text-align:left; color:var(--neon-blue); border-bottom:2px solid rgba(59,130,246,0.2);">
                            <th style="padding:1rem;">League</th>
                            <th style="padding:1rem;">Style</th>
                            <th style="padding:1rem;">Rows</th>
                            <th style="padding:1rem;">Corners MAE</th>
                            <th style="padding:1rem;">Cards MAE</th>
                            <th style="padding:1rem;">Goals MAE</th>
                            <th style="padding:1rem;">Corners O9.5 Brier</th>
                            <th style="padding:1rem;">Cards O4.5 Brier</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rows.map(r => {
                            const profile = profiles.profiles?.[r.league] || {};
                            const counts = r.count_metrics || {};
                            const lines = r.line_metrics || {};
                            return `
                                <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                                    <td style="padding:1rem; font-weight:700;">${r.league}</td>
                                    <td style="padding:1rem; color:var(--text-dim);">${profile.style_label || r?.profile?.style_label || 'learned baseline'}</td>
                                    <td style="padding:1rem; font-family:'Roboto Mono';">${r.n_examples}</td>
                                    <td style="padding:1rem; font-family:'Roboto Mono'; color:var(--neon-gold);">${counts.corners_total?.mae ?? 'N/A'}</td>
                                    <td style="padding:1rem; font-family:'Roboto Mono'; color:var(--neon-red);">${counts.cards_total?.mae ?? 'N/A'}</td>
                                    <td style="padding:1rem; font-family:'Roboto Mono'; color:var(--neon-lime);">${counts.goals_total?.mae ?? 'N/A'}</td>
                                    <td style="padding:1rem; font-family:'Roboto Mono';">${lines['corners_over_9.5']?.brier ?? 'N/A'}</td>
                                    <td style="padding:1rem; font-family:'Roboto Mono';">${lines['cards_over_4.5']?.brier ?? 'N/A'}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        `;
    } catch (e) {
        grid.innerHTML = '<div class="loading-state"><p style="color:var(--neon-red);">Market model reports missing. Run the pipeline or scripts/market_count_models.py.</p></div>';
    }
}

async function loadV11Optimization() {
    const grid = document.getElementById('match-grid');
    const title = document.getElementById('league-title');
    const ticketGrid = document.getElementById('ticket-carousel');
    
    title.innerText = "V11 Temporal Hybrid Optimization";
    ticketGrid.innerHTML = '<div class="ticket-placeholder">Neural Temporal Architecture Active: Analyzing sequence momentum.</div>';
    grid.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Synchronizing V11 Model Results...</p></div>';

    try {
        const res = await fetch(`data/v11_hybrid_results.json?t=${Date.now()}`);
        if (!res.ok) throw new Error("No V11 results found");
        
        const data = await res.json();
        let gnn = null;
        try {
            const gnnRes = await fetch(`data/gnn_training_report.json?t=${Date.now()}`);
            if (gnnRes.ok) gnn = await gnnRes.json();
        } catch (e) {
            gnn = null;
        }
        let gnnWalkForward = null;
        try {
            const wfRes = await fetch(`data/gnn_walk_forward_report.json?t=${Date.now()}`);
            if (wfRes.ok) gnnWalkForward = await wfRes.json();
        } catch (e) {
            gnnWalkForward = null;
        }
        const gnnRows = gnn && Array.isArray(gnn.results) ? gnn.results : [];
        const gnnWfByLeague = {};
        if (gnnWalkForward && Array.isArray(gnnWalkForward.results)) {
            gnnWalkForward.results.forEach(row => { gnnWfByLeague[row.league] = row; });
        }
        grid.innerHTML = `
            <div class="audit-container" style="grid-column: 1 / -1; background: rgba(15, 18, 25, 0.7); border-radius: 12px; padding: 2rem; border: 1px solid var(--neon-blue); box-shadow: 0 0 30px rgba(59, 130, 246, 0.1);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2rem;">
                    <div>
                        <h2 style="color:white; margin:0; font-size:1.5rem;">V11 Hybrid Transformer Results</h2>
                        <p style="color:var(--text-dim); margin:0.5rem 0 0 0; font-size:0.9rem;">Validated Performance across ${data.leagues.length} Core Leagues</p>
                    </div>
                    <div style="text-align:right;">
                        <span class="engine-badge" style="background:rgba(59, 130, 246, 0.1); color:var(--neon-blue); font-size:0.8rem; padding:0.5rem 1rem;">TS: ${data.timestamp}</span>
                    </div>
                </div>

                <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:1.5rem; margin-bottom:2.5rem;">
                    <div class="stat-card" style="background:rgba(255,255,255,0.03);">
                        <span class="label">AVG 1X2 ACCURACY</span>
                        <span class="value" style="color:var(--neon-lime);">
                            ${(data.results.reduce((acc, r) => acc + r.hybrid_accuracy, 0) / data.results.length * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="stat-card spotlight" style="background:rgba(59, 130, 246, 0.05);">
                        <span class="label">AVG PERFORMANCE LIFT</span>
                        <span class="value" style="color:var(--neon-blue);">
                            +${(data.results.reduce((acc, r) => acc + r.improvement, 0) / data.results.length).toFixed(1)}%
                        </span>
                    </div>
                </div>

                <table style="width: 100%; border-collapse: collapse; font-family: 'Inter';">
                    <thead>
                        <tr style="text-align: left; color: var(--neon-blue); border-bottom: 2px solid rgba(59, 130, 246, 0.2);">
                            <th style="padding: 1.25rem;">League</th>
                            <th style="padding: 1.25rem;">V8.1 Base</th>
                            <th style="padding: 1.25rem;">V11 Hybrid</th>
                            <th style="padding: 1.25rem;">Val Acc</th>
                            <th style="padding: 1.25rem;">Delta</th>
                            <th style="padding: 1.25rem;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.results.map(r => `
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.03); color: var(--text-bright); transition: background 0.2s;" onmouseover="this.style.background='rgba(255,255,255,0.01)'" onmouseout="this.style.background='transparent'">
                                <td style="padding: 1.25rem; font-weight: 700; display:flex; align-items:center; gap:0.75rem;">
                                    <div style="width:8px; height:8px; border-radius:50%; background:var(--neon-blue); box-shadow:0 0 8px var(--neon-blue);"></div>
                                    ${r.league}
                                </td>
                                <td style="padding: 1.25rem; font-family: 'Roboto Mono'; color: var(--text-dim);">${(r.base_accuracy * 100).toFixed(1)}%</td>
                                <td style="padding: 1.25rem; font-family: 'Roboto Mono'; color: var(--neon-lime); font-weight:700;">${(r.hybrid_accuracy * 100).toFixed(1)}%</td>
                                <td style="padding: 1.25rem; font-family: 'Roboto Mono'; opacity:0.8;">${(r.val_accuracy * 100).toFixed(1)}%</td>
                                <td style="padding: 1.25rem;">
                                    <span style="color:${r.improvement >= 0 ? 'var(--neon-lime)' : 'var(--neon-red)'}; font-weight:600;">
                                        ${r.improvement >= 0 ? '▲' : '▼'} ${Math.abs(r.improvement).toFixed(1)}%
                                    </span>
                                </td>
                                <td style="padding: 1.25rem;"><span class="badge anchor" style="background:rgba(16, 185, 129, 0.1); color:var(--neon-lime); border-color:rgba(16, 185, 129, 0.2);">DEPLOYED</span></td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>

                <div style="margin-top:2rem; padding:1.5rem; background:rgba(59, 130, 246, 0.05); border-radius:8px; border-left:4px solid var(--neon-blue);">
                    <h4 style="color:var(--neon-blue); margin:0 0 0.5rem 0; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px;">Architecture Notes</h4>
                    <p style="color:var(--text-dim); margin:0; font-size:0.85rem; line-height:1.5;">
                        The V11 Hybrid Transformer uses a <strong>Bidirectional LSTM</strong> encoder with <strong>Attention Pooling</strong> to ingest the last 15 match events as a temporal sequence. 
                        This allows the model to detect momentum shifts and "dead-cat bounces" that static embeddings (V9/V10) fail to capture.
                    </p>
                </div>

                ${gnnRows.length ? `
                    <div style="margin-top:2rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
                            <div>
                                <h2 style="color:white; margin:0; font-size:1.25rem;">GNN Residual Gate</h2>
                                <p style="color:var(--text-dim); margin:0.35rem 0 0 0; font-size:0.85rem;">Graph models activate only when market+graph validation improves log-loss or accuracy.</p>
                            </div>
                            <span class="engine-badge" style="background:rgba(16, 185, 129, 0.08); color:var(--neon-lime); font-size:0.75rem; padding:0.45rem 0.8rem;">${gnn.created_at || 'latest'}</span>
                        </div>
                        <table style="width: 100%; border-collapse: collapse; font-family: 'Inter';">
                            <thead>
                                <tr style="text-align: left; color: var(--neon-lime); border-bottom: 2px solid rgba(16, 185, 129, 0.2);">
                                    <th style="padding: 1rem;">League</th>
                                    <th style="padding: 1rem;">Market Acc</th>
                                    <th style="padding: 1rem;">GNN Acc</th>
                                    <th style="padding: 1rem;">Blend Acc</th>
                                    <th style="padding: 1rem;">Logloss Lift</th>
                                    <th style="padding: 1rem;">Weight</th>
                                    <th style="padding: 1rem;">Holdout</th>
                                    <th style="padding: 1rem;">Walk-Fwd</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${gnnRows.map(r => {
                                    const wf = gnnWfByLeague[r.league];
                                    const wfEnabled = wf ? !!wf.enabled : false;
                                    return `
                                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.03); color: var(--text-bright);">
                                            <td style="padding: 1rem; font-weight:700;">${r.league}</td>
                                            <td style="padding: 1rem; font-family:'Roboto Mono'; color:var(--text-dim);">${((r.market_accuracy || 0) * 100).toFixed(1)}%</td>
                                            <td style="padding: 1rem; font-family:'Roboto Mono';">${((r.validation_accuracy || 0) * 100).toFixed(1)}%</td>
                                            <td style="padding: 1rem; font-family:'Roboto Mono'; color:${r.enabled ? 'var(--neon-lime)' : 'var(--text-dim)'};">${((r.blend_accuracy || 0) * 100).toFixed(1)}%</td>
                                            <td style="padding: 1rem; font-family:'Roboto Mono'; color:${(r.blend_logloss_lift || 0) > 0 ? 'var(--neon-lime)' : 'var(--neon-red)'};">${(r.blend_logloss_lift || 0).toFixed(4)}</td>
                                            <td style="padding: 1rem; font-family:'Roboto Mono';">${((r.blend_weight || 0) * 100).toFixed(0)}%</td>
                                            <td style="padding: 1rem;"><span class="badge anchor" style="background:${r.enabled ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.08)'}; color:${r.enabled ? 'var(--neon-lime)' : 'var(--neon-red)'};">${r.enabled ? 'PASS' : 'FAIL'}</span></td>
                                            <td style="padding: 1rem;"><span class="badge anchor" style="background:${wfEnabled ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.08)'}; color:${wfEnabled ? 'var(--neon-lime)' : 'var(--neon-red)'};">${wfEnabled ? 'ACTIVE' : 'BLOCKED'}</span></td>
                                        </tr>
                                    `;
                                }).join('')}
                            </tbody>
                        </table>
                    </div>
                ` : ''}
            </div>
        `;
    } catch (e) {
        grid.innerHTML = `
            <div class="loading-state">
                <p style="color:var(--neon-red);">V11 Optimization Data Missing</p>
                <p style="font-size:0.8rem; color:var(--text-dim);">Please run the V11 Hybrid pipeline to generate validation results.</p>
                <button onclick="launchV11Pipeline()" style="margin-top:1rem; padding:0.5rem 1rem; background:var(--neon-blue); color:white; border:none; border-radius:4px; cursor:pointer;">Run V11 Pipeline</button>
            </div>
        `;
    }
}

async function loadHistory() {
    const grid = document.getElementById('match-grid');
    const title = document.getElementById('league-title');
    title.innerText = "Historical Intelligence Archive";
    grid.innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Querying Historical Database...</p></div>';
    
    const responses = await Promise.all(
        config.leagues.map(async (league) => {
            try {
                const res = await fetch(`data/${league}.json?t=${Date.now()}`);
                if (!res.ok) return [];
                return await res.json();
            } catch (e) {
                return [];
            }
        })
    );
    const allHistory = responses.flat();
    
    // Sort by date (descending)
    allHistory.sort((a, b) => {
        const dateA = a.Date ? a.Date.split('/').reverse().join('') : '0';
        const dateB = b.Date ? b.Date.split('/').reverse().join('') : '0';
        return dateB.localeCompare(dateA);
    });

    renderMatches(allHistory, true);
    loadTickets('global');
}

async function loadLeague(leagueName) {
    const grid = document.getElementById('match-grid');
    const title = document.getElementById('league-title');
    title.innerText = `${leagueName.replace(/([A-Z])/g, ' $1').trim()} Intelligence`;
    
    try {
        // Try Live API first
        const apiRes = await fetch(`${config.apiBase}/api/v5/predictions/${leagueName}?t=${Date.now()}`).catch(() => null);
        let data = [];

        if (apiRes && apiRes.ok) {
            const apiJson = await apiRes.json();
            data = apiJson.data;
        } else {
            // Fallback to static JSON
            const response = await fetch(`data/${leagueName}.json?t=${Date.now()}`);
            if (!response.ok) {
                grid.innerHTML = `<div class="loading-state"><p>No qualifying Sharp Edges found for this league today.</p></div>`;
                updateHeaderStats([]);
                return;
            }
            data = await response.json();
        }
        
        // Apply Display Date Filter
        const displayFilter = document.getElementById('display-date-filter');
        const targetDate = displayFilter ? displayFilter.value : null;
        let filtered = data;
        if (targetDate) {
            const tDate = targetDate.split('T')[0];
            filtered = data.filter(m => {
                let mDate = m.Date || m.date || m.match_date || '';
                if (mDate.includes('/')) {
                    const parts = mDate.split('/');
                    if (parts.length === 3) mDate = `${parts[2]}-${parts[1].padStart(2, '0')}-${parts[0].padStart(2, '0')}`;
                }
                if (mDate.includes('T')) mDate = mDate.split('T')[0];
                
                if (window.filterMode === 'week') {
                    const d1 = new Date(tDate + 'T00:00:00');
                    const d2 = new Date(mDate + 'T00:00:00');
                    const diffTime = d2 - d1;
                    const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24));
                    return diffDays >= 0 && diffDays <= 7;
                }
                return mDate === tDate;
            });
        }

        renderMatches(filtered);
        updateHeaderStats(filtered);
    } catch (err) {
        grid.innerHTML = `<div class="loading-state"><p style="color: var(--neon-red)">Error: Intelligence Stream Interrupted.</p></div>`;
    }
}

function safePct(val) {
    if (val === null || val === undefined || val === "N/A" || isNaN(val)) return "N/A";
    return (parseFloat(val) * 100).toFixed(0) + "%";
}

function trustScore(match) {
    const raw = match?.trust_score ?? match?.['Trust Score'] ?? match?.execution_trust ?? match?.eqi ?? match?.eqi_score ?? 0;
    const numeric = Number.parseFloat(raw);
    if (!Number.isFinite(numeric)) return 1;
    return Math.max(1, Math.min(100, Math.round(numeric)));
}

function trustTier(score) {
    if (score >= 70) return 'HIGH TRUST';
    if (score >= 45) return 'PLAYABLE TRUST';
    if (score >= 25) return 'LOW TRUST';
    return 'WATCHLIST';
}

const DRIVER_LABELS = {
    'Home_Momentum_Spike': '📈 HOME MOMENTUM',
    'Away_Momentum_Spike': '📈 AWAY MOMENTUM',
    'High_Home_Win_Probability': '🔥 HOME DOMINANCE',
    'Home_Win_Lean': '↗️ HOME LEAN',
    'Strong_Away_Dominance': '🔥 AWAY DOMINANCE',
    'Away_Win_Lean': '↗️ AWAY LEAN',
    'Elevated_Draw_Probability': '⚖️ DRAW RISK',
    'Tactical_Stalemate_Risk': '🛡️ TACTICAL LOCK',
    'High_Scoring_Potential_(O2.5)': '⚽ HIGH SCORING',
    'Attacking_Overload_Expected': '⚔️ ATTACK SPIKE',
    'Both_Teams_Scoring_Likelihood': '🎯 BTTS POTENTIAL',
    'Critical_Motivation': '🏆 MAX PRESSURE',
    'Significant_Match_Pressure': '⚠️ HIGH STAKES',
    'Deep_Set-Piece_Signal': '📐 CORNER EDGE',
    'p_d': 'Market Draw Pressure',
    'p_h': 'Home Win Probability',
    'p_a': 'Away Win Probability',
    'momentum_h': 'Home Tactical Momentum',
    'momentum_a': 'Away Tactical Momentum',
    'sot_h': 'Home Shot Volume',
    'sot_a': 'Away Shot Volume',
    'exp_diff': 'xG Dominance Edge',
    'cs_a': 'Away Clean Sheet Prob',
    'cs_h': 'Home Clean Sheet Prob',
    'ga_h': 'Defensive Leakage (H)',
    'ga_a': 'Defensive Leakage (A)',
    'form_a': 'Away League Form',
    'form_h': 'Home League Form',
    'sot_diff': 'SOT Efficiency Gap',
    'p_btts_model': 'Neural BTTS Signal',
    'p_a_scores': 'Away Scoring Probability',
    'momentum_diff': 'Momentum Cross-Over',
    'exp_h': 'Home Expected Goals',
    'exp_a': 'Away Expected Goals',
    'corn_a': 'Away Corner Volume'
};

function renderMatches(matches) {
    const grid = document.getElementById('match-grid');
    grid.innerHTML = '';

    if (!matches || matches.length === 0) {
        grid.innerHTML = '<div class="loading-state"><p>No intelligence found for this segment.</p></div>';
        return;
    }

    // Group by League (V5.6)
    const groups = {};
    matches.forEach(m => {
        const league = m.League || m.league || 'Global Market';
        if (!groups[league]) groups[league] = [];
        groups[league].push(m);
    });

    Object.keys(groups).sort().forEach(leagueName => {
        // Create League Header
        const leagueHeader = document.createElement('div');
        leagueHeader.className = 'league-header-container';
        leagueHeader.style = "grid-column: 1 / -1; margin-top: 2rem; padding: 1rem; background: rgba(255,255,255,0.02); border-radius: 8px; border-left: 4px solid var(--neon-gold); display: flex; align-items: center; justify-content: space-between;";
        leagueHeader.innerHTML = `
            <div style="font-family: 'Outfit'; font-size: 1.2rem; font-weight: 700; color: var(--neon-gold); letter-spacing: 1px;">
                🏆 ${leagueName.toUpperCase()}
            </div>
            <div style="font-size: 0.7rem; color: var(--text-dim);">
                ${groups[leagueName].length} FIXTURES ANALYZED
            </div>
        `;
        grid.appendChild(leagueHeader);

        // Render Matches for this league
        groups[leagueName].sort((a,b) => (a.Time || '').localeCompare(b.Time || '')).forEach(match => {
            const isSingularity = (match.prob >= 0.85 && match.CLD > 0.05);
            const card = document.createElement('div');
            card.className = `match-card ${match.Anchor === 'YES' ? 'anchor' : ''} ${isSingularity ? 'singularity' : ''}`;
            
            const score = trustScore(match);
            const scoreTier = trustTier(score);
            const badgeClass = score >= 70 ? 'anchor' : (score >= 45 ? 'normal' : 'trap');

            // CLD Sentiment Logic
            let cldSentiment = '<span class="badge" style="background:rgba(255,255,255,0.05); color:var(--text-dim)">Neutral Sentiment</span>';
            if (match.CLD > 0.01) {
                cldSentiment = '<span class="badge" style="background:rgba(16,185,129,0.1); color:var(--neon-lime)">⬆ Sharp Backing</span>';
            } else if (match.CLD < -0.01) {
                cldSentiment = '<span class="badge" style="background:rgba(239,68,68,0.1); color:var(--neon-red)">⬇ Market Drift</span>';
            }

            const mm = match.market_model || {};
            const router = match.market_router || {};
            const cornersO95 = mm.corners?.over_9_5;
            const cardsO45 = mm.cards?.over_4_5;
            const styleLabel = mm.style_label || '';
            const routerProb = router.selected_probability;
            const routerBadge = router.high_competitiveness
                ? `<span class="badge" style="background:rgba(255,184,0,0.1); color:var(--neon-gold); border-color:rgba(255,184,0,0.25)" title="${router.reason || 'High-competitiveness market routing'}">COMP ${Math.round((router.entropy || 0) * 100)}</span>`
                : '';
            const situation = match.league_situation || {};
            const motivationScore = Math.max(0, Math.min(100, Number.parseFloat(match.motivation_score ?? situation.match_pressure_score ?? 0) || 0));
            const motivationLabel = match.motivation_label || situation.label || 'Normal table context';
            const motivationBadge = motivationScore >= 35
                ? `<span class="badge" style="background:rgba(255,184,0,0.1); color:var(--neon-gold); border-color:rgba(255,184,0,0.25)" title="${motivationLabel}">MOT ${Math.round(motivationScore)}</span>`
                : '';

            card.innerHTML = `
                <div class="match-header">
                    <div class="teams">
                        <div class="team-container" style="display:flex; align-items:center; justify-content: space-between; gap:15px; width: 100%;">
                            <div class="market-badge primary">${match.primary_market}</div>
                            <div style="display:flex; flex-direction:column; align-items:flex-end; gap:4px;">
                                <div class="trust-badge" title="Trust Score: ${score}/100">
                                    TRUST ${score}
                                </div>
                                <span class="scs-badge" style="font-size:0.5rem; background:rgba(0,243,255,0.1); color:var(--neon-blue); padding:1px 4px; border:1px solid rgba(0,243,255,0.2); border-radius:4px; text-transform:uppercase">${match.scs_label || 'Standard'}</span>
                                ${match.v8_motif ? `<span class="v8-badge" style="font-size:0.5rem; background:rgba(255,0,0,0.2); color:#ff4444; padding:1px 4px; border:1px solid #ff4444; border-radius:4px; font-weight:bold;" title="V8.0 Structural Motif Detected">⚠️ V8 TRAP</span>` : ''}
                            </div>
                        </div>
                        <div class="match-teams">
                            <div class="team">
                                <span class="team-name">${match.Home}</span>
                                <span class="momentum-icon">${match['Momentum H'] || '➖'}</span>
                            </div>
                            <div class="vs">
                                <span>VS</span>
                                <div class="spectral-power" title="V7.0 Relational Centrality">
                                    ⚡ ${Math.round((match.centrality || 0.5) * 100)}
                                </div>
                            </div>
                            <div class="team">
                                <span class="momentum-icon">${match['Momentum A'] || '➖'}</span>
                                <span class="team-name">${match.Away}</span>
                            </div>
                        </div>
                        <div class="kickoff-time" style="font-size:0.7rem; color:var(--text-dim); margin-top:0.5rem; font-family:'Roboto Mono'">
                            📅 ${match.Date || 'Live'} | ⏰ ${match.Time || '00:00'}
                        </div>
                    </div>
                    <div class="badges">
                        ${isSingularity ? '<span class="singularity-badge">🌌 Singularity Level</span>' : ''}
                        <span class="badge ${badgeClass}">${scoreTier}</span>
                        ${motivationBadge}
                        ${routerBadge}
                        ${cldSentiment}
                    </div>
                </div>

                <div class="prediction-box" style="background: rgba(0, 243, 255, 0.05); border-left: 3px solid var(--neon-blue)">
                    <div>
                        <div class="pred-label">SHARP STAKE</div>
                        <div class="pred-value" style="color:var(--neon-gold)"><span class="counter-val" data-target="${match.stake_pct || 0}">0</span>%</div>
                    </div>
                    <div style="text-align: right">
                        <div class="pred-label">TRUST SCORE</div>
                        <div class="pred-value" style="color:var(--neon-blue)"><span class="counter-val" data-target="${score}">0</span></div>
                    </div>
                </div>

                <!-- V5.5 Intelligence Breakdown -->
                <div class="intelligence-breakdown" style="margin-top: 1rem; padding: 0.75rem; background: rgba(255,255,255,0.02); border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.65rem; margin-bottom: 0.5rem; color: var(--text-dim);">
                        <span>MODEL: <span class="counter-val" data-target="${match.trust_breakdown?.model || (match.source_confidence * 65).toFixed(0)}">0</span>%</span>
                        <span>MARKET: <span class="counter-val" data-target="${match.trust_breakdown?.market || (match.source_confidence * 72).toFixed(0)}">0</span>%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--text-dim); margin-bottom: 0.75rem;">
                        <span>STABILITY: <span class="counter-val" data-target="${match.trust_breakdown?.stability || 45}">0</span>%</span>
                        <span>CONTEXT: <span class="counter-val" data-target="${match.trust_breakdown?.context || (match.source_confidence * 80).toFixed(0)}">0</span>%</span>
                    </div>
                    ${(match.rationale && match.rationale.top_drivers) || match.trust_breakdown?.top_drivers ? `
                    <div class="v11-rationale" style="border-top: 1px solid rgba(255,255,255,0.05); padding-top: 0.5rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.3rem;">
                            <div style="font-size: 0.55rem; color: var(--neon-violet); text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">🧠 V11 Neural Rationale</div>
                            <div style="font-size: 0.5rem; color: var(--text-dim); opacity: 0.7;">CONF: ${match.rationale?.confidence || 'MED'}</div>
                        </div>
                        <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                            ${((match.rationale && match.rationale.top_drivers) || (match.trust_breakdown?.top_drivers || [])).map(d => {
                                const label = DRIVER_LABELS[d.replace(/ /g, '_')] || d.toUpperCase();
                                return `<span style="font-size: 0.55rem; background: rgba(157, 80, 187, 0.1); color: var(--neon-violet); padding: 1px 4px; border-radius: 3px; border: 1px solid rgba(157, 80, 187, 0.2);">${label}</span>`;
                            }).join('')}
                        </div>
                    </div>
                    ` : ''}
                </div>

                <div class="metrics-row" style="margin: 1rem 0; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 1rem">
                    <div class="metric"><span class="val" style="color:var(--neon-lime)">${match.primary_market}</span><span class="lab">Primary</span></div>
                    <div class="metric"><span class="val" style="color:var(--neon-blue)">${match.secondary_market}</span><span class="lab">Secondary</span></div>
                </div>

                <div class="market-section">
                    <!-- V6.0 Quantum Niche Markets -->
                    <div class="niche-metrics" style="display: flex; justify-content: space-between; padding: 0.5rem; background: rgba(255,255,255,0.03); border-radius: 4px; margin-bottom: 0.5rem; font-size: 0.65rem; font-family: 'Roboto Mono';">
                        <span style="color: var(--neon-gold)">🚩 CORNERS: ${match['E(Corners)'] || 'N/A'}</span>
                        <span style="color: var(--neon-red)">🟨 CARDS: ${match['E(Cards)'] || 'N/A'}</span>
                        <span style="color: var(--neon-blue)">👤 ${match['Player Prop'] || 'None'}</span>
                    </div>
                    ${mm && Object.keys(mm).length ? `
                    <div style="display:flex; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.5rem; font-size:0.62rem;">
                        <span style="background:rgba(255,184,0,0.08); color:var(--neon-gold); padding:2px 5px; border-radius:4px;">C O9.5 ${cornersO95 !== undefined ? safePct(cornersO95) : 'N/A'}</span>
                        <span style="background:rgba(255,62,62,0.08); color:var(--neon-red); padding:2px 5px; border-radius:4px;">Y O4.5 ${cardsO45 !== undefined ? safePct(cardsO45) : 'N/A'}</span>
                        ${router && Object.keys(router).length ? `<span style="background:rgba(16,185,129,0.08); color:var(--neon-lime); padding:2px 5px; border-radius:4px;">Router ${routerProb !== undefined && routerProb !== null ? safePct(routerProb) : router.action || 'active'}</span>` : ''}
                        ${styleLabel ? `<span style="background:rgba(0,210,255,0.08); color:var(--neon-blue); padding:2px 5px; border-radius:4px;">${styleLabel}</span>` : ''}
                    </div>
                    ` : ''}

                    <div class="market-grid" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem;">
                        <div class="market-item"><span class="m-label">BTTS</span><span class="m-value">${safePct(match['P(BTTS)'])}</span></div>
                        <div class="market-item"><span class="m-label">O2.5</span><span class="m-value">${safePct(match['P(O2.5)'])}</span></div>
                        <div class="market-item"><span class="m-label">Value</span><span class="m-value">${safePct(match['Value Edge'])}</span></div>
                        <div class="market-item"><span class="m-label">1X</span><span class="m-value">${safePct(match['P(1X)'])}</span></div>
                        <div class="market-item"><span class="m-label">X2</span><span class="m-value">${safePct(match['P(X2)'])}</span></div>
                        <div class="market-item"><span class="m-label">DNB</span><span class="m-value">${safePct(match['DNB'])}</span></div>
                    </div>
                </div>
            `;
            grid.appendChild(card);
        });
    });
    // Trigger high-performance counter animations
    animateCounters();
}
function updateHeaderStats(matches) {
    const total = matches ? matches.length : 0;
    const sharps = matches ? matches.filter(m => trustScore(m) >= 70).length : 0;
    
    document.getElementById('total-games').innerText = total;
    document.getElementById('best-edge').innerText = sharps;
    const bestEdgeLabel = document.getElementById('best-edge-label');
    if (bestEdgeLabel) bestEdgeLabel.innerText = 'High Trust';
}

async function updateLastSyncTime() {
    try {
        const response = await fetch('data/update_log.json');
        const logs = await response.json();
        if (logs.length > 0) {
            document.getElementById('current-date').innerHTML = `
                Last sync | <span style="color:var(--neon-lime)">Active: ${logs[0].timestamp}</span>
            `;
        }
    } catch (e) {}
}

function animateCounters() {
    const counters = document.querySelectorAll('.counter-val');
    counters.forEach(counter => {
        const target = +counter.getAttribute('data-target');
        const speed = 200;
        const updateCount = () => {
            const count = +counter.innerText;
            const inc = target / speed;
            if (count < target) {
                counter.innerText = Math.ceil(count + inc);
                setTimeout(updateCount, 1);
            } else {
                counter.innerText = target;
            }
        };
        updateCount();
    });
}
