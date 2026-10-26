import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Powers of Ten · 70 years of AI models", layout="wide")

HERE = Path(__file__).parent
SAND, BLUE, CORAL, INK, MUTED = "#C8B79E", "#2F6BFF", "#D93A2B", "#111111", "#6B675F"


# =====================================================================
# Helpers
# =====================================================================
def load_css():
    css = (HERE / "style.css").read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


def img(name):
    """Return an image from the images/ folder as a data URI, or None if it's missing."""
    for ext in ("jpg", "jpeg", "png", "webp"):
        path = HERE / "images" / f"{name}.{ext}"
        if path.exists():
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    return None


def photo(name, tint=None, extra=""):
    """A photo frame with an optional era-colour tint. Shows a placeholder if the image is missing."""
    src = img(name)
    inner = f'<img src="{src}" alt="{name}">' if src else f'<div class="ph">add images/{name}.jpg</div>'
    tint_div = f'<div class="tint" style="background:{tint}"></div>' if tint and src else ""
    return f'<div class="photo">{inner}{tint_div}{extra}</div>'


def sup(n):
    """10 to the power n, written with HTML superscript."""
    return f"10<sup>{n}</sup>"


def chapter(num, kicker, title, anchor):
    st.html(f"""
<div class="chapter" id="{anchor}">
  <div class="num">10<sup>{num}</sup></div>
  <div><div class="kicker">{kicker}</div><h2>{title}</h2></div>
</div>""")


def stats(items, n=3):
    cells = "".join(f'<div class="stat"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in items)
    st.html(f'<div class="stats" style="--n:{n}">{cells}</div>')


def pull(text, note=""):
    small = f"<small>{note}</small>" if note else ""
    st.html(f'<div class="pull"><p>{text}</p>{small}</div>')


def style_fig(fig, dark=False, height=440):
    fg = "#EDEAE3" if dark else INK
    grid = "rgba(255,255,255,.08)" if dark else "rgba(0,0,0,.08)"
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=fg, size=13),
        margin=dict(l=70, r=20, t=60, b=60), title_x=0,
        legend=dict(bgcolor="rgba(0,0,0,0)", title_font=dict(size=11)),
        hoverlabel=dict(font_family="JetBrains Mono"),
        title_font_color=fg, legend_font_color=fg,
    )
    fig.update_xaxes(gridcolor=grid, zeroline=False, linecolor=fg, automargin=True)
    fig.update_yaxes(gridcolor=grid, zeroline=False, automargin=True)
    return fig


def fmt_params(x):
    if pd.isna(x):
        return "—"
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"{x / div:.3g}{unit}"
    return f"{x:.0f}"


# =====================================================================
# Data
# =====================================================================
@st.cache_data
def load_data():
    raw = pd.read_csv(HERE / "all_ai_models.csv")
    clean = pd.read_csv(HERE / "clean_data.csv", parse_dates=["date"])
    return raw, clean


@st.cache_resource
def train_model(data):
    features = ["log_parameters", "year"]
    d = data.dropna(subset=features + ["log_compute_flop"])
    X_train, X_test, y_train, y_test = train_test_split(
        d[features], d["log_compute_flop"], test_size=0.2, random_state=42)
    m = LinearRegression().fit(X_train, y_train)
    pred = m.predict(X_test)
    return m, r2_score(y_test, pred), mean_absolute_error(y_test, pred), len(d)


load_css()
raw, df = load_data()

biggest = df.loc[df["parameters"].idxmax()]
first = raw.loc[raw["Model"] == "Theseus"].iloc[0]
raw_missing = raw.isnull().mean().mean()
clean_missing = df.isnull().mean().mean()


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.html("""
<div class="sb-mast"><span>FIELD NOTES</span><span>NO. 001</span></div>
<div class="sb-card">
  <div class="kicker">The dataset</div>
  <h3>Explore the data.</h3>
  <p>Epoch AI's database of AI models, from early experiments to frontier systems, 1950–2026.</p>
  <a href="https://epoch.ai/data/ai-models" target="_blank">Source: Epoch AI ↗</a>
</div>""")

    year_range = st.slider("Year range", int(df["year"].min()), int(df["year"].max()),
                           (2010, int(df["year"].max())))
    domain_options = sorted(df["domain_group"].unique())
    domains = st.pills("Domain", domain_options, selection_mode="multi", default=domain_options)
    org = st.selectbox("Organization type", ["All", "Industry", "Academia", "Collaboration"])

    df_f = df[df["year"].between(*year_range) & df["domain_group"].isin(domains or [])]
    if org != "All":
        df_f = df_f[df_f["org_group"] == org]

    st.html(f"""
<div class="sb-card" style="margin-top:8px">
  <div class="kicker">Models selected</div>
  <div class="sb-count">{len(df_f):,}</div>
  <p style="margin:0">Filters apply to chapter 10<sup>3</sup>.</p>
</div>
<div class="sb-nav">
  <div class="kicker" style="margin-bottom:4px">In this story</div>
  <a href="#ch0">10<sup>0</sup>&nbsp;&nbsp;The big picture</a><br>
  <a href="#ch1">10<sup>1</sup>&nbsp;&nbsp;The raw data</a><br>
  <a href="#ch2">10<sup>2</sup>&nbsp;&nbsp;The clean-up</a><br>
  <a href="#ch3">10<sup>3</sup>&nbsp;&nbsp;What the data shows</a><br>
  <a href="#ch4">10<sup>4</sup>&nbsp;&nbsp;Predict it</a><br>
  <a href="#ch5">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;What we learned</a>
  <div class="sb-rule"></div>
  <p style="font-family:Inter;color:#6B675F;font-size:12px;line-height:1.5">A story about scale, and the work it takes to measure it.</p>
</div>""")


# =====================================================================
# 10^0  Hero: the big picture
# =====================================================================
st.html(f"""
<div class="mast" id="ch0">
  <div class="mast-top"><span>Vol. I · No. 001</span><span>A data story about artificial intelligence</span><span>{pd.Timestamp.today():%A, %d %B %Y}</span></div>
  <div class="mast-title">Powers of Ten</div>
  <div class="mast-bot"><span>70 years of AI models</span><span>Data: Epoch AI</span><span>By Lujain Al Ibrahim</span></div>
</div>

<div class="hero">
  <div>
    <div class="kicker" style="color:{MUTED}">1950—2026 / A data story</div>
    <h1>AI didn't just grow. It grew by <em>powers of ten.</em></h1>
    <p class="lede">From a robot mouse with 40 parameters to systems with over a hundred trillion.
    A journey through seven decades of AI, and the imperfect data behind its rise.</p>
  </div>
  {photo("brain", extra='''
    <div class="overlay"><div class="kicker">Scale comparison</div>
    <p>10<sup>14</sup>: the largest AI model has about as many parameters as there are synapses in a human brain.</p></div>''')}
</div>
""")

strip = [
    ("note", 2, "a sticky note of words", SAND),
    ("city", 6, "the people in a big city", SAND),
    ("crowd", 9, "the people in India", BLUE),
    ("galaxy", 11, "the stars in the Milky Way", BLUE),
    ("neurons", 14, "the synapses in your brain", CORAL),
]
cards = "".join(
    f'<div class="card">{photo(name, tint, f"<div class=exp-badge>{sup(p)}</div>")}<div class="cap">{cap}</div></div>'
    for name, p, cap, tint in strip)
st.html(f'<div class="kicker" style="margin-top:44px;color:{MUTED}">Powers of ten, in things you know</div>'
        f'<div class="strip" style="margin-top:12px">{cards}</div>')

st.html(f"""
<div class="bignum">
  <div class="n">10<sup>14</sup></div>
  <div class="t">parameters in the largest model, <b>{biggest['model']}</b> ({int(biggest['year'])}).
  The first model in the dataset, <b>Theseus</b> ({first['Publication date'][:4]}), a maze-solving robot mouse, had
  <b>{first['Parameters']:.0f}</b>.</div>
</div>

<div class="eras">
  <div class="ends"><span>1950</span><span>→</span><span>2026</span></div>
  <div class="row">
    <div class="era" style="flex:3">{photo("era_early", SAND, '<div class="lbl">Early AI</div>')}
      <div class="bar" style="background:{SAND}"></div><div class="name">Early AI</div><div class="yrs">1950–2009</div></div>
    <div class="era" style="flex:1.6">{photo("era_deep", BLUE, '<div class="lbl">Deep learning</div>')}
      <div class="bar" style="background:{BLUE}"></div><div class="name">Deep learning</div><div class="yrs">2010–2021</div></div>
    <div class="era" style="flex:1.2">{photo("era_scale", CORAL, '<div class="lbl">The scale era</div>')}
      <div class="bar" style="background:{CORAL}"></div><div class="name">The scale era</div><div class="yrs">2022–2026</div></div>
  </div>
</div>
""")


# =====================================================================
# 10^1  The raw data
# =====================================================================
chapter(1, "The raw data / 01", "We started with a mess.", "ch1")
stats([(f"{raw.shape[0]:,}", "Models"), (raw.shape[1], "Columns"), (f"{raw_missing:.0%}", "Of all cells empty")])
st.html('<p class="prose drop">Seventy years of model-building do not arrive in a tidy spreadsheet. Names change. '
        'Fields go missing. The first challenge was not finding the trend: it was making the records comparable.</p>')

# Real rows from the raw file, each showing a different problem
MESSY = [
    ("Theseus", None),
    ("Perceptron Mark I", "org"),
    ("GPT-6.1 Sol", "domain"),
    ("Step 5 Preview", "domain"),
    ("Olmo 3 32B Instruct", "org"),
    ("GigaChat-7B", "date"),
]
NOTES = {None: "", "org": "Academia twice?!", "domain": "same thing, new order", "date": "no date, no domain"}


def cell(v, bad=False, short=None):
    text = "—" if pd.isna(v) else (short or v)
    cls = ' class="bad"' if (bad or pd.isna(v)) else ""
    return f"<td><span{cls}>{text}</span></td>"


rows = ""
for name, issue in MESSY:
    r = raw.loc[raw["Model"] == name].iloc[0]
    org_txt = r["Organization categorization"]
    org_short = (org_txt[:22] + "…") if isinstance(org_txt, str) and len(org_txt) > 22 else org_txt
    note = "empty!" if pd.isna(r["Parameters"]) and not issue else NOTES[issue]
    rows += (f"<tr><td>{name}</td>{cell(r['Publication date'], issue == 'date')}"
             f"{cell(r['Domain'], issue == 'domain')}{cell(r['Parameters'], False, fmt_params(r['Parameters']))}"
             f"{cell(org_txt, issue == 'org', org_short)}<td><span class='note'>{note}</span></td></tr>")

st.html(f"""
<div class="scrap">
  <div class="kicker" style="margin-bottom:10px">Inside the source file: six real rows</div>
  <table class="tbl"><tr><th>Model</th><th>Date</th><th>Domain</th><th>Parameters</th><th>Organization type</th><th></th></tr>{rows}</table>
</div>""")

missing = (raw.isnull().mean() * 100).sort_values(ascending=False).head(20).sort_values()
fig = go.Figure(go.Bar(x=missing.values, y=missing.index, orientation="h",
                       marker_color=[CORAL if v > 90 else SAND for v in missing.values],
                       hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>"))
fig.update_layout(title=dict(text="The 20 emptiest columns  ·  <span style='color:#D93A2B'>red = over 90% empty</span>",
                             font=dict(family="Playfair Display", size=22)))
fig.update_xaxes(title="% missing", range=[0, 100])
st.plotly_chart(style_fig(fig, height=560), width="stretch", theme=None)


# =====================================================================
# 10^2  The clean-up
# =====================================================================
chapter(2, "The clean-up / 02", "Then we cleaned it.", "ch2")


def change(old, new):
    return f'<span class="old">{old}</span><span class="arrow">→</span>{new}'


stats([
    (change(raw.shape[1], df.shape[1]), "Columns"),
    (change(f"{raw_missing:.0%}", f"{clean_missing:.0%}"), "Empty cells"),
    (change(raw["Domain"].nunique(), df["main_domain"].nunique()), "Domain labels"),
])

with st.expander("See every cleaning step"):
    st.markdown("""
1. **Selected 15 useful columns** out of 57. Notes, links and near-empty columns were removed.
2. **Renamed columns** to short names without spaces.
3. **Converted types:** text dates became real dates, and a `year` column was added.
4. **Missing values:** dropped 12 rows with no date, filled categories with "Unknown",
   and left numbers as missing instead of inventing values.
5. **Log columns:** added log10 versions of size, compute and data (4 zero values treated as missing).
6. **Grouped labels:** merged domain and organization combinations into clean groups.
""")

# The same rows, after cleaning
clean_rows = ""
for name, issue in MESSY:
    match = df.loc[df["model"] == name]
    if match.empty:
        clean_rows += f'<tr class="dropped"><td>{name}</td><td colspan="4">dropped: no publication date</td></tr>'
        continue
    r = match.iloc[0]
    clean_rows += (f"<tr><td>{name}</td><td>{r['year']}</td><td><span class='tag'>{r['main_domain']}</span></td>"
                   f"<td><span class='tag'>{r['org_group']}</span></td><td>{fmt_params(r['parameters'])}</td></tr>")
st.html(f"""
<div class="crisp">
  <div class="kicker" style="margin-bottom:10px">The same six rows, after cleaning</div>
  <table class="tbl"><tr><th>Model</th><th>Year</th><th>Domain</th><th>Organization</th><th>Parameters</th></tr>{clean_rows}</table>
</div>""")

left, right = st.columns([1, 1], gap="large")
with left:
    st.html('<h3 style="font-size:28px;margin:0">A log scale makes size readable</h3>'
            '<p class="prose" style="font-size:15px">Flip the switch: on raw numbers, 99.8% of models fall in one bar.</p>')
    scale = st.segmented_control("Scale", ["Raw numbers", "Log10 scale"], default="Log10 scale",
                                 label_visibility="collapsed") or "Log10 scale"
    col = "parameters" if scale == "Raw numbers" else "log_parameters"
    fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=[SAND if col == "parameters" else INK])
    fig.update_xaxes(title="Parameters" if col == "parameters" else "log10(Parameters)  ·  9 = 1 billion")
    fig.update_yaxes(title="Models")
    st.plotly_chart(style_fig(fig, height=360), width="stretch", theme=None)
with right:
    multi_spellings = raw["Domain"].dropna().loc[lambda s: s.str.contains(",")].nunique()
    st.html('<h3 style="font-size:28px;margin:0">Cleaning revealed a hidden giant</h3>'
            f'<p class="prose" style="font-size:15px">Multimodal models were split across '
            f'<b>{multi_spellings}</b> different spellings. Grouped, they are the 2nd-largest domain.</p>')
    counts = df["main_domain"].value_counts().head(8).sort_values()
    fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h",
                           marker_color=[CORAL if d == "Multimodal" else SAND for d in counts.index],
                           text=counts.values, textposition="outside",
                           hovertemplate="%{y}: %{x} models<extra></extra>"))
    fig.update_xaxes(title="Models", range=[0, counts.max() * 1.15])
    st.plotly_chart(style_fig(fig, height=360), width="stretch", theme=None)


# =====================================================================
# 10^3  What the data shows  (navy: the page gets darker as AI gets bigger)
# =====================================================================
with st.container(key="sec_navy"):
    chapter(3, "What the data shows / 03", "Bigger, faster, and built by industry.", "ch3")

    if df_f.empty:
        st.warning("No models match these filters. Try widening them in the sidebar.")
    else:
        stats([(f"{len(df_f):,}", "Models selected"),
               (fmt_params(df_f["parameters"].median()), "Median size (parameters)"),
               (f"{(df_f['org_group'] == 'Industry').mean():.0%}", "Built by industry")])

        palette = {"Language": CORAL, "Multimodal": "#E9A097", "Vision": BLUE,
                   "Biology": "#2E8B6E", "Image generation": "#C49A1E", "Other": "#A39E92"}
        fig = px.scatter(df_f.dropna(subset=["log_parameters"]), x="date", y="log_parameters",
                         color="domain_group", color_discrete_map=palette, hover_name="model",
                         hover_data={"organization": True, "date": "|%Y", "log_parameters": ":.1f",
                                     "domain_group": False},
                         labels={"log_parameters": "log10(Parameters)  ·  9 = 1 billion",
                                 "date": "", "domain_group": "Domain"})
        fig.update_traces(marker=dict(size=6, opacity=0.75, line=dict(width=0)))
        for start, end, colour, label in [("1950", "2010", SAND, "EARLY AI"),
                                          ("2010", "2022", BLUE, "DEEP LEARNING"),
                                          ("2022", "2027", CORAL, "SCALE ERA")]:
            fig.add_vrect(x0=f"{start}-01-01", x1=f"{end}-01-01", fillcolor=colour, opacity=0.10, line_width=0,
                          annotation_text=label, annotation_position="top left",
                          annotation_font=dict(size=10, color=colour, family="JetBrains Mono"))
        lo, hi = df_f["date"].min(), df_f["date"].max()
        fig.update_xaxes(range=[lo - pd.Timedelta(days=200), hi + pd.Timedelta(days=200)])
        fig.update_layout(title=dict(text="Model size over time", font=dict(family="Playfair Display", size=24)))
        st.plotly_chart(style_fig(fig, dark=False, height=520), width="stretch", theme=None)
        pull("Before 2010, model size doubled roughly every four years. After 2010, it doubled roughly every twelve months.",
             "Linear fit on log10(parameters) by publication date")

        per_year = (df_f[df_f["org_group"].isin(["Industry", "Academia", "Collaboration"])]
                    .groupby(["year", "org_group"]).size().reset_index(name="models"))
        fig = px.bar(per_year, x="year", y="models", color="org_group", barmode="group",
                     color_discrete_map={"Industry": CORAL, "Academia": BLUE, "Collaboration": SAND},
                     labels={"year": "", "models": "Models released", "org_group": "Built by"})
        fig.update_layout(title=dict(text="Who builds AI?", font=dict(family="Playfair Display", size=24)))
        st.plotly_chart(style_fig(fig, dark=False, height=440), width="stretch", theme=None)
        pull("A typical industry model is about 70× bigger than an academic one.",
             "Median parameters, models from 2010 onwards. Note: 2025–2026 are still incomplete.")

        with st.expander("Explore the filtered data"):
            st.dataframe(df_f[["model", "organization", "country", "year", "main_domain",
                               "org_group", "parameters", "compute_flop"]], width="stretch")
            st.dataframe(df_f[["parameters", "compute_flop", "dataset_size", "year"]].describe(), width="stretch")


# =====================================================================
# 10^4  Predict it  (black: the scale era)
# =====================================================================
with st.container(key="sec_black"):
    chapter(4, "Predict it / 04", "How much compute did it take?", "ch4")
    st.html('<p class="prose">Many labs hide how much compute they used, but they often reveal model size. '
            'A linear regression, trained on models that report both, estimates compute from size and year.</p>')

    model, r2, mae, n = train_model(df)

    controls, gauge = st.columns([1, 1.3], gap="large")
    with controls:
        size_b = st.number_input("Model size (billions of parameters)", min_value=0.001,
                                 max_value=100000.0, value=70.0, step=1.0, format="%.1f")
        year_in = st.slider("Release year", 2010, 2026, 2024)
        X_new = pd.DataFrame({"log_parameters": [np.log10(size_b * 1e9)], "year": [year_in]})
        log_pred = float(model.predict(X_new)[0])
        mant, expo = 10 ** (log_pred % 1), int(log_pred // 1)
        st.html(f"""
<div class="readout">
  <div class="kicker">Predicted training compute</div>
  <div class="val">{mant:.1f}×10<sup>{expo}</sup></div>
  <div class="sub">FLOP · trained on <b>{n:,}</b> models · R² <b>{r2:.2f}</b> · typical error <b>~{10 ** mae:.0f}×</b></div>
</div>""")
    with gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge", value=log_pred,
            gauge=dict(
                axis=dict(range=[0, 27], tickvals=list(range(0, 28, 3)),
                          ticktext=[f"10<sup>{t}</sup>" for t in range(0, 28, 3)],
                          tickfont=dict(family="JetBrains Mono", size=11, color=MUTED)),
                bar=dict(color=CORAL, thickness=0.28),
                bgcolor="#EAE4D8", borderwidth=0,
                steps=[dict(range=[0, 15], color="#E6DCC9"), dict(range=[15, 22], color="#DCE3F5"),
                       dict(range=[22, 27], color="#F3D9D4")],
                threshold=dict(line=dict(color=INK, width=3), thickness=0.9, value=log_pred))))
        fig.add_annotation(x=0.5, y=0.18, showarrow=False, text="log10 FLOP",
                           font=dict(family="JetBrains Mono", size=12, color=MUTED))
        fig = style_fig(fig, dark=False, height=380)
        fig.update_layout(margin=dict(l=50, r=50, t=40, b=10))
        st.plotly_chart(fig, width="stretch", theme=None)


# =====================================================================
# What we learned  (your words go here)
# =====================================================================
st.html('<div class="chapter" id="ch5" style="border-top:0"><div class="num" style="font-size:80px">∑</div>'
        '<div><div class="kicker">What we learned / 05</div><h2>Four things the data told us.</h2></div></div>')

# ✏️ Replace each placeholder with your own insight: (big number, headline, one sentence)
# Key takeaways: (big number, headline, sentence)
TAKEAWAYS = [
    ("12 months", "AI is growing faster than ever",
     "Before 2010, the size of AI models doubled about every four years. Since then, it has doubled roughly every 12 months."),
    ("70×", "Industry builds the biggest models",
     "A typical model from industry is about 70 times larger than one from a university, because only companies can afford the computing power that large models require."),
    ("191→22", "Cleaning uncovered a hidden giant",
     "Merging 191 inconsistent labels into 22 revealed that multimodal models, scattered across 170 different spellings, form the second-largest domain."),
    ("R² 0.85", "A model's cost can be estimated",
     "Knowing only a model's size and release year, a simple regression explains 85% of the differences in training compute, typically within a factor of 10 to 11."),
]
st.html('<div class="takeaways">' + "".join(
    f'<div class="t"><div class="no">0{i}</div><div class="big">{big}</div><h4>{head}</h4><p>{body}</p></div>'
    for i, (big, head, body) in enumerate(TAKEAWAYS, 1)) + "</div>")

# Limitations
st.html('<div class="limits"><div class="kicker" style="color:#111">Limitations</div>'
        'Many values are missing, especially parameters and training compute, because companies increasingly keep them secret, so the largest recent models are likely underestimated. Data for 2025–2026 is still incomplete, and the prediction model learned from only the 1,253 models that report both size and compute.</div>')

st.html("""
<div class="footer">
  <span>Data: Epoch AI, <i>Data on AI Models</i> · epoch.ai/data/ai-models</span>
  <span>Dashboard by Lujain Al Ibrahim</span>
</div>""")
