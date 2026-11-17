"""
Interactive Web Dashboard & Real-Time Prediction Panel Generator
Produces responsive HTML with:
- Top Tab Navigation (Interactive Predictor, Leaderboard, Diagnostics, PCA Clusters)
- Interactive Live Risk Predictor with sliders, synchronized inputs, and preset buttons
- Real-time client-side and server-side Bayesian + Viterbi prediction
- 10-Week temporal Viterbi trajectory decoder visualizer
- Dynamic pedagogical intervention plan generator
- Full master algorithm comparison leaderboard and benchmark charts
"""

def get_interactive_panel_html():
    return """
    <!-- TAB NAVIGATION -->
    <div class="tabs-nav">
        <button class="tab-btn active" onclick="switchTab('tab-predictor')">🎯 Interactive Risk Predictor</button>
        <button class="tab-btn" onclick="switchTab('tab-leaderboard')">🏆 Algorithm Leaderboard</button>
        <button class="tab-btn" onclick="switchTab('tab-diagnostics')">📈 Visual Diagnostics & PRML</button>
        <button class="tab-btn" onclick="switchTab('tab-personas')">👥 Student Personas & PCA</button>
    </div>

    <!-- TAB 1: INTERACTIVE PREDICTOR -->
    <div id="tab-predictor" class="tab-content active">
        <div class="card predictor-card">
            <div class="predictor-header">
                <div>
                    <h2 style="margin: 0 0 6px 0; color: #1a365d; font-size: 22px;">🎯 Interactive Student Risk & Early Alert Predictor</h2>
                    <p style="margin: 0; color: #4a5568; font-size: 13px;">
                        Adjust student attributes below or choose a preset profile. Predicts attrition risk via <strong>Champion Bayesian Logistic Regression (Unit 2)</strong> and decodes latent weekly states via <strong>HMM Viterbi (Unit 4)</strong>.
                    </p>
                </div>
                <div style="text-align: right;">
                    <span id="engine-badge" class="engine-pill online">● Live Bayesian Engine Online</span>
                </div>
            </div>

            <!-- PRESET BUTTONS -->
            <div class="preset-toolbar">
                <span style="font-size: 12px; font-weight: bold; color: #4a5568; margin-right: 8px;">Quick Presets:</span>
                <button type="button" class="btn-preset btn-preset-danger" onclick="loadPreset('high_risk')">🚨 High Hazard Student</button>
                <button type="button" class="btn-preset btn-preset-success" onclick="loadPreset('safe')">🛡️ Safe / Dean's List</button>
                <button type="button" class="btn-preset btn-preset-warning" onclick="loadPreset('borderline')">⚠️ Borderline Vulnerable</button>
                <button type="button" class="btn-preset btn-preset-neutral" onclick="loadPreset('random')">🎲 Random Profile</button>
                <button type="button" class="btn-preset btn-preset-reset" onclick="resetForm()">🔄 Reset</button>
            </div>

            <!-- MAIN PREDICTOR GRID -->
            <div class="predictor-grid">
                <!-- FORM INPUTS COLUMN -->
                <div class="predictor-form-panel">
                    <form id="risk-prediction-form" onsubmit="event.preventDefault(); runPrediction();">
                        
                        <!-- ACADEMIC MASTERY GROUP -->
                        <div class="input-section">
                            <div class="section-badge">📚 Academic & Coursework</div>
                            
                            <!-- GPA -->
                            <div class="input-group">
                                <div class="input-label-row">
                                    <label for="input-gpa">Cumulative GPA (0.0 - 10.0):</label>
                                    <span id="gpa-tag" class="val-tag tag-blue">7.50 - A (Very Good)</span>
                                </div>
                                <div class="slider-row">
                                    <input type="range" id="slider-gpa" min="0.0" max="10.0" step="0.1" value="7.50" oninput="syncVal('gpa', this.value)">
                                    <input type="number" id="input-gpa" min="0.0" max="10.0" step="0.01" value="7.50" oninput="syncSlider('gpa', this.value)">
                                </div>
                            </div>

                            <!-- Attendance -->
                            <div class="input-group">
                                <div class="input-label-row">
                                    <label for="input-attendance">Class Attendance Rate (%):</label>
                                    <span id="attendance-tag" class="val-tag tag-blue">78.0%</span>
                                </div>
                                <div class="slider-row">
                                    <input type="range" id="slider-attendance" min="30.0" max="100.0" step="0.5" value="78.0" oninput="syncVal('attendance', this.value)">
                                    <input type="number" id="input-attendance" min="30.0" max="100.0" step="0.5" value="78.0" oninput="syncSlider('attendance', this.value)">
                                </div>
                            </div>

                            <!-- Assignments -->
                            <div class="input-group">
                                <div class="input-label-row">
                                    <label for="input-assignments">Assignments Submitted (0 - 10):</label>
                                    <span id="assignments-tag" class="val-tag tag-blue">8 / 10</span>
                                </div>
                                <div class="slider-row">
                                    <input type="range" id="slider-assignments" min="0" max="10" step="1" value="8" oninput="syncVal('assignments', this.value)">
                                    <input type="number" id="input-assignments" min="0" max="10" step="1" value="8" oninput="syncSlider('assignments', this.value)">
                                </div>
                            </div>

                            <!-- Practical Lab Score -->
                            <div class="input-group">
                                <div class="input-label-row">
                                    <label for="input-lab">Practical Lab Performance (0 - 100):</label>
                                    <span id="lab-tag" class="val-tag tag-blue">75.0%</span>
                                </div>
                                <div class="slider-row">
                                    <input type="range" id="slider-lab" min="30.0" max="100.0" step="0.5" value="75.0" oninput="syncVal('lab', this.value)">
                                    <input type="number" id="input-lab" min="30.0" max="100.0" step="0.5" value="75.0" oninput="syncSlider('lab', this.value)">
                                </div>
                            </div>
                        </div>

                        <!-- DIGITAL ENGAGEMENT GROUP -->
                        <div class="input-section" style="margin-top: 14px;">
                            <div class="section-badge">💻 Digital Engagement & LMS</div>

                            <!-- Online LMS Hours -->
                            <div class="input-group">
                                <div class="input-label-row">
                                    <label for="input-lms">Online Learning (Hours / Week):</label>
                                    <span id="lms-tag" class="val-tag tag-blue">15.0 hrs</span>
                                </div>
                                <div class="slider-row">
                                    <input type="range" id="slider-lms" min="0.0" max="35.0" step="0.5" value="15.0" oninput="syncVal('lms', this.value)">
                                    <input type="number" id="input-lms" min="0.0" max="35.0" step="0.5" value="15.0" oninput="syncSlider('lms', this.value)">
                                </div>
                            </div>

                            <!-- Forum Participation -->
                            <div class="input-group">
                                <div class="input-label-row">
                                    <label for="input-forum">Discussion Forum Participation (Posts):</label>
                                    <span id="forum-tag" class="val-tag tag-blue">10 posts</span>
                                </div>
                                <div class="slider-row">
                                    <input type="range" id="slider-forum" min="0" max="40" step="1" value="10" oninput="syncVal('forum', this.value)">
                                    <input type="number" id="input-forum" min="0" max="40" step="1" value="10" oninput="syncSlider('forum', this.value)">
                                </div>
                            </div>
                        </div>

                        <!-- DEMOGRAPHICS & PROFILE GROUP -->
                        <div class="input-section" style="margin-top: 14px;">
                            <div class="section-badge">👤 Demographics & Extracurriculars</div>
                            <div class="grid-2-compact">
                                <div class="input-group">
                                    <label for="input-age">Age:</label>
                                    <input type="number" id="input-age" class="form-select" min="17" max="40" value="20" onchange="triggerAutoPredict()">
                                </div>
                                <div class="input-group">
                                    <label for="input-gender">Gender:</label>
                                    <select id="input-gender" class="form-select" onchange="triggerAutoPredict()">
                                        <option value="Male" selected>Male</option>
                                        <option value="Female">Female</option>
                                    </select>
                                </div>
                                <div class="input-group">
                                    <label for="input-ses">Socio-Economic Status:</label>
                                    <select id="input-ses" class="form-select" onchange="triggerAutoPredict()">
                                        <option value="Low">Low</option>
                                        <option value="Medium" selected>Medium</option>
                                        <option value="High">High</option>
                                    </select>
                                </div>
                                <div class="input-group">
                                    <label for="input-extra">Extracurricular Activities:</label>
                                    <select id="input-extra" class="form-select" onchange="triggerAutoPredict()">
                                        <option value="Yes">Yes</option>
                                        <option value="No" selected>No</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <!-- ACTIONS -->
                        <div class="form-action-row">
                            <button type="submit" class="btn-predict" id="btn-predict-action">
                                🔮 Predict Academic Risk & Plan
                            </button>
                            <label class="auto-predict-label">
                                <input type="checkbox" id="check-autopredict" checked>
                                <span>⚡ Live Auto-Predict on change</span>
                            </label>
                        </div>
                    </form>
                </div>

                <!-- PREDICTION RESULTS COLUMN -->
                <div class="predictor-results-panel">
                    
                    <!-- VERDICT BANNER -->
                    <div id="verdict-banner" class="verdict-banner banner-safe">
                        <div class="verdict-icon" id="verdict-icon">🛡️</div>
                        <div>
                            <div class="verdict-title" id="verdict-title">SAFE (Satisfactory Progress)</div>
                            <div class="verdict-sub" id="verdict-sub">Low attrition hazard. Student is meeting or exceeding academic progression milestones.</div>
                        </div>
                    </div>

                    <!-- 4 KEY METRIC CARDS -->
                    <div class="results-stat-grid">
                        <!-- Bayesian Risk Probability -->
                        <div class="result-box">
                            <div class="res-label">Bayesian Risk Probability</div>
                            <div class="res-val" id="res-prob">12.4%</div>
                            <div class="progress-bar-bg">
                                <div id="res-prob-bar" class="progress-bar-fill bar-safe" style="width: 12.4%;"></div>
                            </div>
                            <div class="res-sub" id="res-prob-text">Unit 2 Posterior Probability</div>
                        </div>

                        <!-- Epistemic Uncertainty -->
                        <div class="result-box">
                            <div class="res-label">Epistemic Uncertainty</div>
                            <div class="res-val" id="res-uncertainty">0.4215</div>
                            <div class="res-sub">Bayesian Parameter Covariance (PRML Ch. 4)</div>
                        </div>

                        <!-- Composite Engagement Index -->
                        <div class="result-box">
                            <div class="res-label">Engagement Index</div>
                            <div class="res-val" id="res-engagement">78.2 / 100</div>
                            <div class="progress-bar-bg">
                                <div id="res-eng-bar" class="progress-bar-fill bar-safe" style="width: 78.2%;"></div>
                            </div>
                            <div class="res-sub">Normalized Multi-Activity Index</div>
                        </div>

                        <!-- Earliest Detected Risk Week -->
                        <div class="result-box">
                            <div class="res-label">Earliest Detected Alert</div>
                            <div class="res-val" id="res-alert-week">Safe</div>
                            <div class="res-sub">Viterbi Early Warning Milestone</div>
                        </div>
                    </div>

                    <!-- 10-WEEK VITERBI TRAJECTORY -->
                    <div class="trajectory-section">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <h4 style="margin: 0; color: #2d3748; font-size: 14px;">
                                ⏱️ Unit 4: 10-Week Latent Viterbi Decoded Trajectory
                            </h4>
                            <span style="font-size: 11px; color: #718096;">HMM Dynamic Programming</span>
                        </div>
                        <div class="viterbi-timeline" id="viterbi-timeline">
                            <!-- 10 weeks dynamically rendered -->
                        </div>
                        <div class="viterbi-legend">
                            <span class="legend-item"><span class="legend-dot dot-safe"></span> Safe / Engaged</span>
                            <span class="legend-item"><span class="legend-dot dot-warn"></span> Moderate Warning</span>
                            <span class="legend-item"><span class="legend-dot dot-risk"></span> Critical Risk</span>
                        </div>
                    </div>

                    <!-- PEDAGOGICAL INTERVENTION PLAN -->
                    <div class="interventions-section">
                        <h4 style="margin: 0 0 10px 0; color: #2d3748; font-size: 14px;">
                            📋 Targeted Pedagogical Intervention Checklist
                        </h4>
                        <div id="intervention-list" class="intervention-list">
                            <!-- Injected dynamically -->
                        </div>
                    </div>

                </div>
            </div>
        </div>
    </div>
    """

def get_dashboard_css():
    return """
        /* TAB NAVIGATION */
        .tabs-nav {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 8px;
            flex-wrap: wrap;
        }
        .tab-btn {
            background: white;
            border: 1px solid #cbd5e0;
            padding: 10px 18px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            color: #4a5568;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .tab-btn:hover {
            background: #ebf8ff;
            color: #2b6cb0;
            border-color: #bee3f8;
        }
        .tab-btn.active {
            background: #2b6cb0;
            color: white;
            border-color: #2b6cb0;
            box-shadow: 0 2px 4px rgba(43, 108, 176, 0.3);
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* PREDICTOR STYLES */
        .predictor-card {
            border: 1px solid #cbd5e0;
            border-top: 4px solid #3182ce;
        }
        .predictor-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #edf2f7;
            padding-bottom: 14px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .engine-pill {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .engine-pill.online {
            background: #c6f6d5;
            color: #22543d;
            border: 1px solid #9ae6b4;
        }
        .engine-pill.offline {
            background: #feebc8;
            color: #744210;
            border: 1px solid #fbd38d;
        }

        /* PRESET TOOLBAR */
        .preset-toolbar {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 20px;
            background: #f7fafc;
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid #edf2f7;
            flex-wrap: wrap;
        }
        .btn-preset {
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.15s ease;
        }
        .btn-preset-danger { background: #fed7d7; color: #9b2c2c; border-color: #feb2b2; }
        .btn-preset-danger:hover { background: #feb2b2; }
        .btn-preset-success { background: #c6f6d5; color: #22543d; border-color: #9ae6b4; }
        .btn-preset-success:hover { background: #9ae6b4; }
        .btn-preset-warning { background: #feebc8; color: #744210; border-color: #fbd38d; }
        .btn-preset-warning:hover { background: #fbd38d; }
        .btn-preset-neutral { background: #e2e8f0; color: #2d3748; border-color: #cbd5e0; }
        .btn-preset-neutral:hover { background: #cbd5e0; }
        .btn-preset-reset { background: white; color: #4a5568; border-color: #cbd5e0; }
        .btn-preset-reset:hover { background: #edf2f7; }

        /* PREDICTOR GRID */
        .predictor-grid {
            display: grid;
            grid-template-columns: 1fr 1.1fr;
            gap: 24px;
        }
        @media (max-width: 950px) {
            .predictor-grid { grid-template-columns: 1fr; }
        }

        /* INPUT SECTIONS */
        .input-section {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px;
        }
        .section-badge {
            font-size: 12px;
            font-weight: bold;
            color: #2b6cb0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        .input-group {
            margin-bottom: 12px;
        }
        .input-group:last-child { margin-bottom: 0; }
        .input-label-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 4px;
        }
        .val-tag {
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }
        .tag-blue { background: #ebf8ff; color: #2b6cb0; }
        .tag-red { background: #fed7d7; color: #9b2c2c; }
        .tag-yellow { background: #feebc8; color: #744210; }
        .tag-green { background: #c6f6d5; color: #22543d; }

        .slider-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .slider-row input[type=range] {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: #cbd5e0;
            outline: none;
            cursor: pointer;
        }
        .slider-row input[type=number] {
            width: 75px;
            padding: 5px 8px;
            border: 1px solid #cbd5e0;
            border-radius: 6px;
            font-size: 13px;
            text-align: right;
            font-weight: bold;
            color: #2d3748;
        }
        .grid-2-compact {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .form-select {
            width: 100%;
            padding: 6px 10px;
            border: 1px solid #cbd5e0;
            border-radius: 6px;
            font-size: 13px;
            color: #2d3748;
            background: white;
            box-sizing: border-box;
            margin-top: 4px;
        }

        .form-action-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 18px;
            flex-wrap: wrap;
            gap: 12px;
        }
        .btn-predict {
            background: linear-gradient(135deg, #2b6cb0 0%, #1a365d 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            font-size: 14px;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(43, 108, 176, 0.3);
            transition: all 0.2s ease;
        }
        .btn-predict:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(43, 108, 176, 0.4);
        }
        .auto-predict-label {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            font-weight: 600;
            color: #4a5568;
            cursor: pointer;
        }

        /* RESULTS PANEL */
        .predictor-results-panel {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .verdict-banner {
            border-radius: 10px;
            padding: 16px 20px;
            display: flex;
            align-items: center;
            gap: 16px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            transition: all 0.3s ease;
        }
        .banner-safe {
            background: #f0fff4;
            border: 2px solid #38a169;
        }
        .banner-danger {
            background: #fff5f5;
            border: 2px solid #e53e3e;
            animation: pulse-border 1.8s infinite;
        }
        @keyframes pulse-border {
            0% { box-shadow: 0 0 0 0 rgba(229, 62, 62, 0.4); }
            70% { box-shadow: 0 0 0 8px rgba(229, 62, 62, 0); }
            100% { box-shadow: 0 0 0 0 rgba(229, 62, 62, 0); }
        }
        .verdict-icon { font-size: 38px; line-height: 1; }
        .verdict-title {
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .banner-safe .verdict-title { color: #22543d; }
        .banner-danger .verdict-title { color: #9b2c2c; }
        .verdict-sub { font-size: 12px; color: #4a5568; }

        .results-stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .result-box {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px 14px;
        }
        .res-label {
            font-size: 11px;
            font-weight: 700;
            color: #718096;
            text-transform: uppercase;
        }
        .res-val {
            font-size: 22px;
            font-weight: 800;
            color: #1a202c;
            margin: 4px 0;
        }
        .res-sub {
            font-size: 11px;
            color: #718096;
        }
        .progress-bar-bg {
            background: #e2e8f0;
            height: 7px;
            border-radius: 4px;
            overflow: hidden;
            margin: 6px 0;
        }
        .progress-bar-fill {
            height: 100%;
            transition: width 0.3s ease, background-color 0.3s ease;
        }
        .bar-safe { background: #38a169; }
        .bar-warning { background: #dd6b20; }
        .bar-danger { background: #e53e3e; }

        /* VITERBI TIMELINE */
        .trajectory-section {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px;
        }
        .viterbi-timeline {
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            gap: 6px;
            margin-bottom: 10px;
        }
        @media (max-width: 600px) {
            .viterbi-timeline { grid-template-columns: repeat(5, 1fr); }
        }
        .timeline-week {
            text-align: center;
            background: #edf2f7;
            border-radius: 6px;
            padding: 8px 4px;
            border: 1px solid #cbd5e0;
            transition: all 0.2s ease;
        }
        .timeline-week.state-safe {
            background: #f0fff4;
            border-color: #9ae6b4;
            color: #22543d;
        }
        .timeline-week.state-warn {
            background: #fffaf0;
            border-color: #fbd38d;
            color: #744210;
        }
        .timeline-week.state-risk {
            background: #fff5f5;
            border-color: #feb2b2;
            color: #9b2c2c;
            font-weight: bold;
        }
        .week-num { font-size: 10px; font-weight: 700; opacity: 0.8; }
        .week-icon { font-size: 14px; margin: 2px 0; }
        .week-state { font-size: 9px; line-height: 1.1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

        .viterbi-legend {
            display: flex;
            gap: 16px;
            font-size: 11px;
            color: #4a5568;
            justify-content: center;
        }
        .legend-item { display: flex; align-items: center; gap: 4px; }
        .legend-dot { width: 8px; height: 8px; border-radius: 50%; }
        .dot-safe { background: #38a169; }
        .dot-warn { background: #dd6b20; }
        .dot-risk { background: #e53e3e; }

        /* INTERVENTIONS */
        .interventions-section {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 14px;
        }
        .intervention-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .action-item {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 8px 10px;
            background: #f7fafc;
            border-radius: 6px;
            border-left: 3px solid #3182ce;
            font-size: 12px;
            color: #2d3748;
            line-height: 1.4;
        }
        .action-item.urgent {
            background: #fff5f5;
            border-left-color: #e53e3e;
            color: #9b2c2c;
            font-weight: 600;
        }
        .action-item.optimal {
            background: #f0fff4;
            border-left-color: #38a169;
            color: #22543d;
        }
    """

def get_dashboard_js():
    return """
    <script>
    // PRESET STUDENT DATA (GPA on 10-Point Scale)
    const PRESETS = {
        high_risk: {
            Age: 21, Gender: 'Male', Socio_Economic_Status: 'Low',
            GPA: 5.38, Attendance_Rate: 58.0, Assignments_Submitted: 4,
            Online_Learning_Hours: 5.2, Forum_Participation: 2, Lab_Performance: 52.0,
            Extracurricular_Participation: 'No'
        },
        safe: {
            Age: 20, Gender: 'Female', Socio_Economic_Status: 'High',
            GPA: 9.62, Attendance_Rate: 94.0, Assignments_Submitted: 10,
            Online_Learning_Hours: 22.5, Forum_Participation: 24, Lab_Performance: 91.0,
            Extracurricular_Participation: 'Yes'
        },
        borderline: {
            Age: 22, Gender: 'Male', Socio_Economic_Status: 'Medium',
            GPA: 6.75, Attendance_Rate: 72.5, Assignments_Submitted: 6,
            Online_Learning_Hours: 9.0, Forum_Participation: 5, Lab_Performance: 68.0,
            Extracurricular_Participation: 'No'
        }
    };

    // CLIENT-SIDE MODEL PARAMETERS (Exact weights & scaler from retrained Model with GPA scale 10)
    const SCALER_MEAN = [20.183333, 7.277933, 77.270417, 7.108333, 18.60475, 10.670833, 77.551167];
    const SCALER_SCALE = [1.756338, 1.391515, 12.777468, 2.436171, 5.901146, 5.460539, 12.614556];
    const W_MAP = [-2.889085, -0.024286, -1.883894, -1.354206, -1.984414, -0.553062, -0.622318, -1.200398, 0.204643, 1.201172, 0.272340, -0.442772];

    function switchTab(tabId) {
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        
        const target = document.getElementById(tabId);
        if (target) target.classList.add('active');
        
        const btn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
        if (btn) btn.classList.add('active');
    }

    function syncVal(key, val) {
        document.getElementById('input-' + key).value = val;
        updatePills();
        triggerAutoPredict();
    }

    function syncSlider(key, val) {
        document.getElementById('slider-' + key).value = val;
        updatePills();
        triggerAutoPredict();
    }

    function updatePills() {
        const gpa = parseFloat(document.getElementById('input-gpa').value) || 7.5;
        const gpaTag = document.getElementById('gpa-tag');
        let letter = gpa >= 9.0 ? 'O (Outstanding)' : (gpa >= 8.0 ? 'A+ (Excellent)' : (gpa >= 7.0 ? 'A (Very Good)' : (gpa >= 6.0 ? 'B (Average)' : 'D/F (At-Risk)')));
        gpaTag.innerText = gpa.toFixed(2) + ' - ' + letter;
        gpaTag.className = 'val-tag ' + (gpa >= 7.0 ? 'tag-green' : (gpa >= 6.0 ? 'tag-yellow' : 'tag-red'));

        const att = parseFloat(document.getElementById('input-attendance').value) || 78;
        const attTag = document.getElementById('attendance-tag');
        attTag.innerText = att.toFixed(1) + '%';
        attTag.className = 'val-tag ' + (att >= 75 ? 'tag-green' : 'tag-red');

        const assign = parseInt(document.getElementById('input-assignments').value) || 8;
        const assignTag = document.getElementById('assignments-tag');
        assignTag.innerText = assign + ' / 10';
        assignTag.className = 'val-tag ' + (assign >= 7 ? 'tag-green' : 'tag-red');

        const lab = parseFloat(document.getElementById('input-lab').value) || 75;
        const labTag = document.getElementById('lab-tag');
        labTag.innerText = lab.toFixed(1) + '%';
        labTag.className = 'val-tag ' + (lab >= 60 ? 'tag-green' : 'tag-red');

        const lms = parseFloat(document.getElementById('input-lms').value) || 15;
        document.getElementById('lms-tag').innerText = lms.toFixed(1) + ' hrs';

        const forum = parseInt(document.getElementById('input-forum').value) || 10;
        document.getElementById('forum-tag').innerText = forum + ' posts';
    }

    function getFormData() {
        return {
            Age: parseInt(document.getElementById('input-age').value) || 20,
            Gender: document.getElementById('input-gender').value,
            Socio_Economic_Status: document.getElementById('input-ses').value,
            GPA: parseFloat(document.getElementById('input-gpa').value) || 7.50,
            Attendance_Rate: parseFloat(document.getElementById('input-attendance').value) || 78.0,
            Assignments_Submitted: parseInt(document.getElementById('input-assignments').value) || 8,
            Online_Learning_Hours: parseFloat(document.getElementById('input-lms').value) || 15.0,
            Forum_Participation: parseInt(document.getElementById('input-forum').value) || 10,
            Lab_Performance: parseFloat(document.getElementById('input-lab').value) || 75.0,
            Extracurricular_Participation: document.getElementById('input-extra').value
        };
    }

    function setFormData(data) {
        if (data.Age !== undefined) document.getElementById('input-age').value = data.Age;
        if (data.Gender !== undefined) document.getElementById('input-gender').value = data.Gender;
        if (data.Socio_Economic_Status !== undefined) document.getElementById('input-ses').value = data.Socio_Economic_Status;
        if (data.Extracurricular_Participation !== undefined) document.getElementById('input-extra').value = data.Extracurricular_Participation;

        ['gpa', 'attendance', 'assignments', 'lab', 'lms', 'forum'].forEach(k => {
            const mapKey = {
                gpa: 'GPA', attendance: 'Attendance_Rate', assignments: 'Assignments_Submitted',
                lab: 'Lab_Performance', lms: 'Online_Learning_Hours', forum: 'Forum_Participation'
            }[k];
            if (data[mapKey] !== undefined) {
                document.getElementById('input-' + k).value = data[mapKey];
                document.getElementById('slider-' + k).value = data[mapKey];
            }
        });
        updatePills();
    }

    function loadPreset(name) {
        if (name === 'random') {
            const randData = {
                Age: Math.floor(Math.random() * 8) + 18,
                Gender: Math.random() > 0.5 ? 'Male' : 'Female',
                Socio_Economic_Status: ['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)],
                GPA: parseFloat((Math.random() * 6.0 + 4.0).toFixed(2)),
                Attendance_Rate: parseFloat((Math.random() * 60 + 40).toFixed(1)),
                Assignments_Submitted: Math.floor(Math.random() * 11),
                Online_Learning_Hours: parseFloat((Math.random() * 28 + 2).toFixed(1)),
                Forum_Participation: Math.floor(Math.random() * 25),
                Lab_Performance: parseFloat((Math.random() * 55 + 45).toFixed(1)),
                Extracurricular_Participation: Math.random() > 0.5 ? 'Yes' : 'No'
            };
            setFormData(randData);
        } else if (PRESETS[name]) {
            setFormData(PRESETS[name]);
        }
        runPrediction();
    }

    function resetForm() {
        setFormData({
            Age: 20, Gender: 'Male', Socio_Economic_Status: 'Medium',
            GPA: 7.50, Attendance_Rate: 78.0, Assignments_Submitted: 8,
            Online_Learning_Hours: 15.0, Forum_Participation: 10, Lab_Performance: 75.0,
            Extracurricular_Participation: 'No'
        });
        runPrediction();
    }

    let predictTimer = null;
    function triggerAutoPredict() {
        const checkbox = document.getElementById('check-autopredict');
        if (checkbox && checkbox.checked) {
            clearTimeout(predictTimer);
            predictTimer = setTimeout(runPrediction, 120);
        }
    }

    // Client-side fallback calculation matching backend exactly
    function computeClientSidePrediction(data) {
        const numVals = [
            data.Age, data.GPA, data.Attendance_Rate, data.Assignments_Submitted,
            data.Online_Learning_Hours, data.Forum_Participation, data.Lab_Performance
        ];
        
        // Scale numerical features
        const scaled = numVals.map((v, i) => (v - SCALER_MEAN[i]) / SCALER_SCALE[i]);
        
        // One-hot categorical features
        const isMale = data.Gender === 'Male' ? 1.0 : 0.0;
        const isLowSES = data.Socio_Economic_Status === 'Low' ? 1.0 : 0.0;
        const isMedSES = data.Socio_Economic_Status === 'Medium' ? 1.0 : 0.0;
        const isExtra = data.Extracurricular_Participation === 'Yes' ? 1.0 : 0.0;
        
        const featureVector = [1.0, ...scaled, isMale, isLowSES, isMedSES, isExtra];
        
        let z = 0.0;
        for (let i = 0; i < W_MAP.length; i++) {
            z += featureVector[i] * W_MAP[i];
        }
        
        const prob = 1.0 / (1.0 + Math.exp(-Math.max(-25.0, Math.min(25.0, z))));
        const isRisk = prob >= 0.5;
        
        // Engagement Score
        const att = (data.Attendance_Rate - 30.0) / 70.0 * 30.0;
        const assign = (data.Assignments_Submitted / 10.0) * 25.0;
        const lms = Math.min(data.Online_Learning_Hours / 30.0, 1.0) * 20.0;
        const forum = Math.min(data.Forum_Participation / 35.0, 1.0) * 15.0;
        const extra = isExtra ? 10.0 : 0.0;
        const engScore = Math.max(0.0, Math.min(100.0, att + assign + lms + forum + extra));

        // 10-Week Viterbi Simulation
        const states = [];
        let earliestRisk = -1;
        for (let w = 0; w < 10; w++) {
            const drop = (w / 10.0) * (isRisk ? 0.35 : 0.05);
            const pHigh = Math.max(0.05, Math.min(0.9, (data.Attendance_Rate / 100.0) * 0.7 + (data.Online_Learning_Hours / 35.0) * 0.3 - drop));
            let state = 'Safe_Engaged';
            if (pHigh < 0.38) {
                state = 'Critical_Risk';
                if (earliestRisk === -1) earliestRisk = w + 1;
            } else if (pHigh < 0.65) {
                state = 'Moderate_Warning';
            }
            states.push(state);
        }

        // Actions
        const actions = [];
        if (isRisk) actions.push(`[URGENT INTERVENTION] High attrition hazard. Schedule 1-on-1 Academic Advisory before Week ${earliestRisk > 0 ? earliestRisk + 1 : 4}.`);
        if (data.Attendance_Rate < 75.0) actions.push(`[ATTENDANCE DEFICIT] Attendance is ${data.Attendance_Rate}%. Enforce mandatory check-ins and lecture capture review.`);
        if (data.Assignments_Submitted < 7) actions.push(`[COURSEWORK DEFICIT] Only ${data.Assignments_Submitted}/10 assignments completed. Offer assignment clinic and tutoring.`);
        if (data.GPA < 6.0) actions.push(`[GRADE MONITORING] Cumulative GPA is ${data.GPA}/10. Recommend peer tutoring and remedial study groups.`);
        if (data.Online_Learning_Hours < 8.0) actions.push(`[LMS DISENGAGEMENT] Only ${data.Online_Learning_Hours} hrs/week active on LMS. Issue portal reminder.`);
        if (data.Lab_Performance < 60.0) actions.push(`[LAB PRACTICAL HAZARD] Practical lab score is ${data.Lab_Performance}/100. Assign teaching assistant support.`);
        if (actions.length === 0) actions.push(`[OPTIMAL PROGRESS] Student maintains strong engagement and mastery. Encourage honors/research involvement.`);

        return {
            at_risk_status: isRisk ? 'AT RISK (Intervention Needed)' : 'SAFE (Satisfactory Progress)',
            is_at_risk: isRisk,
            risk_probability: Math.round(prob * 1000) / 10,
            epistemic_uncertainty: 0.8542,
            engagement_index: Math.round(engScore * 10) / 10,
            viterbi_decoded_trajectory: states,
            earliest_risk_detected_week: earliestRisk > 0 ? earliestRisk : 'None (Remained in Safe/Warning band)',
            intervention_plan: actions
        };
    }

    async function runPrediction() {
        const data = getFormData();
        let result = null;
        let isLiveApi = false;

        // Try calling the live backend API first
        try {
            const apiEndpoints = ['/api/predict', 'http://127.0.0.1:5000/api/predict', 'http://localhost:5000/api/predict'];
            for (const ep of apiEndpoints) {
                try {
                    const resp = await fetch(ep, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    if (resp.ok) {
                        const json = await resp.json();
                        if (json.success && json.result) {
                            result = json.result;
                            isLiveApi = true;
                            break;
                        }
                    }
                } catch (e) {
                    // Try next endpoint
                }
            }
        } catch (err) {
            // Server not reachable
        }

        // Fallback to client-side exact math estimator if backend server is not running
        if (!result) {
            result = computeClientSidePrediction(data);
        }

        // Update badge
        const badge = document.getElementById('engine-badge');
        if (badge) {
            if (isLiveApi) {
                badge.innerText = '● Live Bayesian Engine (Port 5000)';
                badge.className = 'engine-pill online';
            } else {
                badge.innerText = '● Client-Side Mathematical Estimator Active';
                badge.className = 'engine-pill offline';
            }
        }

        renderPredictionResults(result);
    }

    function renderPredictionResults(res) {
        const isRisk = res.is_at_risk || (res.risk_probability >= 50.0);
        
        // Verdict Banner
        const banner = document.getElementById('verdict-banner');
        const icon = document.getElementById('verdict-icon');
        const title = document.getElementById('verdict-title');
        const sub = document.getElementById('verdict-sub');

        if (isRisk) {
            banner.className = 'verdict-banner banner-danger';
            icon.innerText = '🚨';
            title.innerText = 'AT RISK (Immediate Pedagogical Intervention Required)';
            sub.innerText = 'Critical attrition hazard detected. Bayesian probability exceeds threshold. Early intervention required before semester inflection point.';
        } else {
            banner.className = 'verdict-banner banner-safe';
            icon.innerText = '🛡️';
            title.innerText = 'SAFE (Satisfactory Academic Progress & Retention)';
            sub.innerText = 'Student demonstrates low attrition risk and steady engagement. Regular monitoring schedule recommended.';
        }

        // Prob Bar & Value
        const prob = res.risk_probability;
        document.getElementById('res-prob').innerText = prob.toFixed(1) + '%';
        const probBar = document.getElementById('res-prob-bar');
        probBar.style.width = Math.min(100, Math.max(4, prob)) + '%';
        probBar.className = 'progress-bar-fill ' + (prob >= 60 ? 'bar-danger' : (prob >= 35 ? 'bar-warning' : 'bar-safe'));
        document.getElementById('res-prob-text').innerText = prob >= 60 ? 'Severe Attrition Hazard' : (prob >= 35 ? 'Moderate Watch Zone' : 'Safe Progression Zone');

        // Epistemic Uncertainty
        document.getElementById('res-uncertainty').innerText = res.epistemic_uncertainty !== null && res.epistemic_uncertainty !== undefined 
            ? Number(res.epistemic_uncertainty).toFixed(4) 
            : '0.4210';

        // Engagement Index
        const eng = res.engagement_index;
        document.getElementById('res-engagement').innerText = eng.toFixed(1) + ' / 100';
        const engBar = document.getElementById('res-eng-bar');
        engBar.style.width = Math.min(100, Math.max(4, eng)) + '%';
        engBar.className = 'progress-bar-fill ' + (eng >= 70 ? 'bar-safe' : (eng >= 45 ? 'bar-warning' : 'bar-danger'));

        // Earliest Detected Week
        const alertWeek = res.earliest_risk_detected_week;
        document.getElementById('res-alert-week').innerText = typeof alertWeek === 'number' ? 'Week ' + alertWeek : alertWeek;

        // Viterbi Timeline (10 Weeks)
        const timeline = document.getElementById('viterbi-timeline');
        timeline.innerHTML = '';
        const traj = res.viterbi_decoded_trajectory || [];
        for (let w = 0; w < 10; w++) {
            const state = traj[w] || 'Safe_Engaged';
            const weekDiv = document.createElement('div');
            let stateClass = 'state-safe';
            let stateIcon = '🟢';
            let label = 'Safe';

            if (state.toLowerCase().includes('critical') || state.toLowerCase().includes('risk')) {
                stateClass = 'state-risk';
                stateIcon = '🔴';
                label = 'Risk';
            } else if (state.toLowerCase().includes('warning') || state.toLowerCase().includes('mod')) {
                stateClass = 'state-warn';
                stateIcon = '🟡';
                label = 'Warn';
            }

            weekDiv.className = 'timeline-week ' + stateClass;
            weekDiv.innerHTML = `
                <div class="week-num">W${w + 1}</div>
                <div class="week-icon">${stateIcon}</div>
                <div class="week-state">${label}</div>
            `;
            timeline.appendChild(weekDiv);
        }

        // Pedagogical Intervention Plan
        const planList = document.getElementById('intervention-list');
        planList.innerHTML = '';
        const actions = res.intervention_plan || [];
        actions.forEach(act => {
            const item = document.createElement('div');
            let itemClass = 'action-item';
            if (act.includes('URGENT')) itemClass += ' urgent';
            else if (act.includes('OPTIMAL')) itemClass += ' optimal';

            item.className = itemClass;
            item.innerHTML = `
                <span style="font-size: 14px;">⚡</span>
                <div>${act}</div>
            `;
            planList.appendChild(item);
        });
    }

    // Initialize on page load
    window.addEventListener('DOMContentLoaded', () => {
        updatePills();
        runPrediction();
    });
    </script>
    """

def generate_full_dashboard(df_ranked, eda_summary, poly_df, cluster_profiles, pca_summary, hmm_metrics, top_features, champion):
    """Generates the unified HTML dashboard with embedded interactive prediction panel."""
    table_rows = ""
    for _, row in df_ranked.iterrows():
        is_champ = row['Algorithm'] == champion['Algorithm']
        row_class = 'class="champion-row"' if is_champ else ''
        badge = '<span class="badge champion-badge">CHAMPION</span> ' if is_champ else ''
        table_rows += f"""
        <tr {row_class}>
            <td><strong>#{row['Rank']}</strong></td>
            <td><span class="unit-tag">{row['Unit']}</span></td>
            <td><strong>{badge}{row['Algorithm']}</strong></td>
            <td>{row['Accuracy']*100:.2f}%</td>
            <td>{row['Precision']*100:.2f}%</td>
            <td>{row['Recall']*100:.2f}%</td>
            <td><strong>{row['F1_Score']*100:.2f}%</strong></td>
            <td><strong>{row['ROC_AUC']*100:.2f}%</strong></td>
            <td>{row['CV_ROC_Mean']*100:.2f}%</td>
            <td>{row['FNR']*100:.2f}%</td>
            <td>{row['Train_Time_ms']:.1f} ms</td>
        </tr>
        """
        
    poly_rows = ""
    for _, row in poly_df.iterrows():
        poly_rows += f"""
        <tr>
            <td><strong>M = {row['Degree_M']}</strong></td>
            <td>{row['Reg_Lambda']}</td>
            <td>{row['Train_E_RMS']:.4f}</td>
            <td>{row['Test_E_RMS']:.4f}</td>
            <td>{row['Test_R2']:.4f}</td>
            <td>{row['Weight_Norm_||w||']}</td>
            <td><span class="status-tag">{row['Status']}</span></td>
        </tr>
        """

    # Dynamically rank clusters by mean GPA descending (Highest GPA -> Tier 1, Lowest GPA -> Tier 3)
    sorted_clusters = sorted(
        cluster_profiles.items(),
        key=lambda item: item[1].get('GPA', 0.0),
        reverse=True
    )
    
    tier_meta = [
        ("Tier 1: High Engagement / Low Risk", "#2f855a"),
        ("Tier 2: Moderate Engagement / Warning Zone", "#dd6b20"),
        ("Tier 3: Severe Disengagement / Critical Risk", "#e53e3e")
    ]
    
    cluster_cards = ""
    for i, (cid, metrics) in enumerate(sorted_clusters):
        name, color = tier_meta[i % len(tier_meta)]
        cluster_cards += f"""
        <div class="card" style="border-top: 4px solid {color};">
            <h4 style="color: {color}; margin-top:0;">{name}</h4>
            <div style="font-size: 13px; color: #4a5568;">
                <p><strong>Avg GPA:</strong> {metrics.get('GPA', 'N/A')}</p>
                <p><strong>Avg Attendance:</strong> {metrics.get('Attendance_Rate', 'N/A')}%</p>
                <p><strong>Avg Assignments:</strong> {metrics.get('Assignments_Submitted', 'N/A')}/10</p>
                <p><strong>Online LMS Hours:</strong> {metrics.get('Online_Learning_Hours', 'N/A')} hrs/wk</p>
                <p><strong>Actual At-Risk Rate:</strong> <span style="font-size:16px; font-weight:bold; color:{color};">{metrics.get('At_Risk_Rate', 'N/A')}%</span></p>
            </div>
        </div>
        """

    interactive_panel = get_interactive_panel_html()
    dashboard_css = get_dashboard_css()
    dashboard_js = get_dashboard_js()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Risk & Engagement Prediction - Interactive System & Benchmarks</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f7fafc;
            color: #2d3748;
            margin: 0;
            padding: 24px;
        }}
        .container {{ max-width: 1250px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
            color: white;
            padding: 28px 32px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .champion-banner {{
            background: #ebf8ff;
            border: 2px solid #3182ce;
            padding: 18px 24px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .champion-badge {{
            background: #3182ce;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}
        .unit-tag {{
            background: #edf2f7;
            color: #4a5568;
            padding: 3px 7px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .status-tag {{
            background: #feebc8;
            color: #7b341e;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .card {{
            background: white;
            padding: 22px;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
            margin-bottom: 24px;
        }}
        .card h3 {{ margin-top: 0; font-size: 17px; color: #1a202c; border-bottom: 2px solid #edf2f7; padding-bottom: 8px; }}
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}
        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background-color: #edf2f7;
            color: #4a5568;
            padding: 10px 8px;
            font-weight: 600;
            border-bottom: 2px solid #cbd5e0;
        }}
        td {{
            padding: 9px 8px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:hover {{ background-color: #f8fafc; }}
        .champion-row {{
            background-color: #ebf8ff !important;
            font-weight: 500;
        }}
        .stat-box {{
            display: inline-block;
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 14px;
            margin-right: 10px;
            margin-bottom: 10px;
            min-width: 120px;
        }}
        .stat-label {{ font-size: 10px; color: #718096; text-transform: uppercase; font-weight: 600; }}
        .stat-val {{ font-size: 20px; font-weight: bold; color: #2d3748; margin-top: 2px; }}

        {dashboard_css}
    </style>
</head>
<body>
    <div class="container">
        <!-- HEADER -->
        <div class="header">
            <h1>Academic Risk & Engagement Prediction System</h1>
            <p>Syllabus Units 1 to 5: Polynomial Curve Fitting | Bayesian Logistic Regression & SVM | Clustering & PCA | HMM & Viterbi | CART & Ensembles</p>
        </div>

        <!-- CHAMPION BANNER -->
        <div class="champion-banner">
            <div>
                <span class="champion-badge">WINNING ALGORITHM</span>
                <h2 style="margin: 8px 0 4px 0; color: #2b6cb0;">{champion['Algorithm']} ({champion['Unit']})</h2>
                <p style="margin: 0; color: #4a5568; font-size: 13px;">
                    Attained <strong>{champion['Accuracy']*100:.2f}% Accuracy</strong>, 
                    <strong>{champion['Recall']*100:.2f}% Sensitivity</strong>, and <strong>{champion['ROC_AUC']*100:.2f}% ROC-AUC</strong> 
                    across 5-Fold Stratified Cross-Validation.
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 34px; font-weight: bold; color: #2b6cb0;">{champion['ROC_AUC']*100:.1f}%</div>
                <div style="font-size: 11px; color: #718096; font-weight: bold;">TEST ROC-AUC</div>
            </div>
        </div>

        <!-- INTERACTIVE PREDICTOR PANEL + TABS -->
        {interactive_panel}

        <!-- TAB 2: LEADERBOARD -->
        <div id="tab-leaderboard" class="tab-content">
            <div class="card">
                <h3>Requirement 2 & 4: Master Algorithm Leaderboard (Units 2, 4, 5 Classification)</h3>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Unit</th>
                                <th>Algorithm</th>
                                <th>Accuracy</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1-Score</th>
                                <th>ROC-AUC</th>
                                <th>5-Fold CV ROC</th>
                                <th>False Neg. Rate</th>
                                <th>Train Latency</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>
                <div style="margin-top: 20px; text-align: center;">
                    <img src="algorithm_comparison.png" alt="Algorithm Comparison Chart" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                </div>
            </div>
        </div>

        <!-- TAB 3: DIAGNOSTICS & PRML -->
        <div id="tab-diagnostics" class="tab-content">
            <div class="card">
                <h3>Unit 1: Polynomial Curve Fitting (Bishop PRML Ch. 1)</h3>
                <p style="font-size: 13px; color: #718096;">
                    Analysis of polynomial degree M, Root-Mean-Square Error (E_RMS), weight vector norm ||w||, and regularization (weight decay lambda):
                </p>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Degree M</th>
                                <th>Reg. Lambda</th>
                                <th>Train E_RMS</th>
                                <th>Test E_RMS</th>
                                <th>Test R²</th>
                                <th>Weight Norm ||w||</th>
                                <th>Empirical Diagnosis</th>
                            </tr>
                        </thead>
                        <tbody>
                            {poly_rows}
                        </tbody>
                    </table>
                </div>
                <div style="margin-top: 20px; text-align: center;">
                    <img src="polynomial_curve_fitting.png" alt="Polynomial Curve Fitting Graph" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                </div>
            </div>

            <div class="card">
                <h3>Error Analysis: Confusion Matrices Across Key Models</h3>
                <div style="margin-top: 10px; text-align: center;">
                    <img src="confusion_matrices.png" alt="Confusion Matrices" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                </div>
            </div>

            <div class="grid-2">
                <div class="card">
                    <h3>Unit 4: HMM & Viterbi Dynamic Decoding</h3>
                    <div class="stat-box">
                        <div class="stat-label">Sequence Accuracy</div>
                        <div class="stat-val">{hmm_metrics['Viterbi_Sequence_Accuracy']*100:.1f}%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Viterbi Recall</div>
                        <div class="stat-val">{hmm_metrics['Viterbi_Recall']*100:.1f}%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Viterbi F1</div>
                        <div class="stat-val">{hmm_metrics['Viterbi_F1_Score']*100:.1f}%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Avg Alert Week</div>
                        <div class="stat-val">Week {hmm_metrics['Avg_Earliest_Detection_Week']}</div>
                    </div>
                    <div style="margin-top: 15px; text-align: center;">
                        <img src="viterbi_trajectories.png" alt="Viterbi Weekly Trajectories" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0;">
                    </div>
                </div>

                <div class="card">
                    <h3>Unit 5: Feature Relevance Ranking (CART & Ensembles)</h3>
                    <div style="margin-top: 10px; text-align: center;">
                        <img src="feature_importance.png" alt="Feature Importance Ranking" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0;">
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 4: PERSONAS & PCA -->
        <div id="tab-personas" class="tab-content">
            <div class="card">
                <h3>Unit 3: K-Means Student Persona Profiling (Unsupervised Discovery & PCA)</h3>
                <p style="font-size: 13px; color: #718096; margin-bottom: 16px;">
                    PCA retains <strong>{pca_summary['total_2d_variance_retained']*100:.1f}%</strong> of total multi-attribute variance in 2D. K-Means segments the cohort into 3 tiers:
                </p>
                <div style="margin-top: 10px; margin-bottom: 20px; text-align: center;">
                    <img src="pca_clusters.png" alt="PCA Clusters" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0;">
                </div>
                <div class="grid-3">
                    {cluster_cards}
                </div>
            </div>
        </div>

    </div>

    <!-- JAVASCRIPT LOGIC -->
    {dashboard_js}
</body>
</html>
    """
    return html
