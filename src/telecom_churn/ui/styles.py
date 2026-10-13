"""Design system and custom CSS for Orange Telecom Churn Dashboard."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --brand-orange: #FF7900;
    --brand-orange-light: #FFA144;
    --brand-orange-dark: #D45D00;
    --bg-dark: #0D1117;
    --surface-dark: #161B22;
    --surface-border: rgba(255, 121, 0, 0.2);
    --text-primary: #F0F6FC;
    --text-secondary: #8B949E;
    --success: #238636;
    --warning: #D29922;
    --danger: #DA3633;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(255,121,0,0.15) 0%, rgba(13,17,23,0.85) 100%);
    border: 1px solid rgba(255,121,0,0.3);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.hero-title {
    color: #FFFFFF;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hero-title span {
    color: var(--brand-orange);
}

.hero-subtitle {
    color: #A0AEC0;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.5;
}

/* KPI Metric Cards */
.kpi-card {
    background: rgba(22, 27, 34, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 20px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}

.kpi-card:hover {
    transform: translateY(-3px);
    border-color: rgba(255, 121, 0, 0.4);
    box-shadow: 0 8px 24px rgba(255, 121, 0, 0.15);
}

.kpi-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 2.0rem;
    font-weight: 700;
    color: var(--text-primary);
}

.kpi-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 6px;
}

/* Risk Badges */
.badge-low {
    background: rgba(35, 134, 54, 0.2);
    color: #3FB950;
    border: 1px solid rgba(63, 185, 80, 0.3);
}

.badge-medium {
    background: rgba(210, 153, 34, 0.2);
    color: #E3B341;
    border: 1px solid rgba(227, 179, 65, 0.3);
}

.badge-high {
    background: rgba(218, 54, 51, 0.2);
    color: #F85149;
    border: 1px solid rgba(248, 81, 73, 0.3);
}

/* Insight Cards */
.insight-box {
    background: rgba(22, 27, 34, 0.8);
    border-left: 4px solid var(--brand-orange);
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.95rem;
    color: #E6EDF3;
}

.rec-box {
    background: rgba(35, 134, 54, 0.1);
    border-left: 4px solid #3FB950;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.92rem;
    color: #E6EDF3;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 12px 20px;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-secondary);
}

.stTabs [aria-selected="true"] {
    color: var(--brand-orange) !important;
    border-bottom: 2px solid var(--brand-orange) !important;
    background: rgba(255, 121, 0, 0.05);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #FF7900 0%, #D45D00 100%);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(255, 121, 0, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(255, 121, 0, 0.45);
    color: white;
}
</style>
"""
