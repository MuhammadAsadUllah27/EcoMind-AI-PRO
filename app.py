import gradio as gr
import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import datetime

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[EcoMind] Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# CLIP  –  ViT-Large for best zero-shot accuracy
# ─────────────────────────────────────────────────────────────────────────────
CLIP_MODEL_ID  = "openai/clip-vit-large-patch14"
clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)

# ─────────────────────────────────────────────────────────────────────────────
# GENERATIVE MODEL  –  Flan-T5 Large
# ─────────────────────────────────────────────────────────────────────────────
GEN_MODEL_ID = "google/flan-t5-large"
tokenizer    = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
gen_model    = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_ID)

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORIES  (12)
# ─────────────────────────────────────────────────────────────────────────────
CATEGORIES = [
    {
        "name":        "Plastic Bottle / Container",
        "key":         "plastic",
        "icon":        "♻️",
        "color":       "#3B82F6",
        "impact":      7,          # environmental impact score 1-10 (10 = worst)
        "recyclable":  True,
        "hazardous":   False,
        "carbon_kg":   1.9,        # kg CO₂ per kg if landfilled
        "value_usd":   0.25,       # $/kg scrap value (approx.)
        "decompose":   "450 years",
        "prompts": [
            "a plastic bottle",
            "plastic packaging waste",
            "PET plastic container",
            "disposable plastic water bottle",
        ],
    },
    {
        "name":        "Metal Can / Tin",
        "key":         "metal",
        "icon":        "⚙️",
        "color":       "#6B7280",
        "impact":      5,
        "recyclable":  True,
        "hazardous":   False,
        "carbon_kg":   2.5,
        "value_usd":   0.55,
        "decompose":   "80–200 years",
        "prompts": [
            "an aluminium can",
            "steel tin can",
            "metal food can",
            "beverage can waste",
        ],
    },
    {
        "name":        "Paper / Cardboard",
        "key":         "paper",
        "icon":        "📄",
        "color":       "#F59E0B",
        "impact":      3,
        "recyclable":  True,
        "hazardous":   False,
        "carbon_kg":   1.1,
        "value_usd":   0.12,
        "decompose":   "2–6 weeks",
        "prompts": [
            "paper waste cardboard",
            "newspaper waste",
            "cardboard box",
            "crumpled paper trash",
        ],
    },
    {
        "name":        "Glass Bottle / Jar",
        "key":         "glass",
        "icon":        "🔷",
        "color":       "#10B981",
        "impact":      4,
        "recyclable":  True,
        "hazardous":   False,
        "carbon_kg":   0.9,
        "value_usd":   0.08,
        "decompose":   "1 million years",
        "prompts": [
            "a glass bottle",
            "broken glass waste",
            "glass jar",
            "glass container garbage",
        ],
    },
    {
        "name":        "Food / Organic Waste",
        "key":         "organic",
        "icon":        "🌱",
        "color":       "#84CC16",
        "impact":      4,
        "recyclable":  True,
        "hazardous":   False,
        "carbon_kg":   0.5,
        "value_usd":   0.05,
        "decompose":   "2–5 weeks",
        "prompts": [
            "food waste vegetable scraps",
            "organic kitchen waste",
            "rotten fruit food waste",
            "compostable organic matter",
        ],
    },
    {
        "name":        "E-Waste / Electronics",
        "key":         "ewaste",
        "icon":        "⚡",
        "color":       "#8B5CF6",
        "impact":      9,
        "recyclable":  True,
        "hazardous":   True,
        "carbon_kg":   5.8,
        "value_usd":   2.00,
        "decompose":   "100–1000 years",
        "prompts": [
            "electronic waste circuit board",
            "broken laptop phone e-waste",
            "old computer parts waste",
            "discarded electronics gadgets",
        ],
    },
    {
        "name":        "Textiles / Clothes",
        "key":         "textile",
        "icon":        "👕",
        "color":       "#EC4899",
        "impact":      6,
        "recyclable":  True,
        "hazardous":   False,
        "carbon_kg":   3.6,
        "value_usd":   0.40,
        "decompose":   "20–200 years",
        "prompts": [
            "old clothes textile waste",
            "worn out fabric garments",
            "discarded clothing waste",
            "textile rags fabric scraps",
        ],
    },
    {
        "name":        "Hazardous / Chemical Waste",
        "key":         "hazardous",
        "icon":        "☢️",
        "color":       "#EF4444",
        "impact":      10,
        "recyclable":  False,
        "hazardous":   True,
        "carbon_kg":   6.5,
        "value_usd":   0.00,
        "decompose":   "Unknown / permanent",
        "prompts": [
            "hazardous chemical waste drum",
            "toxic waste container",
            "chemical drum barrel hazardous",
            "industrial chemical waste",
        ],
    },
    {
        "name":        "Battery / Cell",
        "key":         "battery",
        "icon":        "🔋",
        "color":       "#F97316",
        "impact":      9,
        "recyclable":  True,
        "hazardous":   True,
        "carbon_kg":   4.2,
        "value_usd":   0.90,
        "decompose":   "100 years",
        "prompts": [
            "used batteries waste",
            "dead battery disposal",
            "lithium battery waste",
            "old AA battery waste",
        ],
    },
    {
        "name":        "Rubber / Tyre",
        "key":         "rubber",
        "icon":        "🛞",
        "color":       "#78716C",
        "impact":      7,
        "recyclable":  True,
        "hazardous":   False,
        "carbon_kg":   2.8,
        "value_usd":   0.30,
        "decompose":   "50–80 years",
        "prompts": [
            "old rubber tyre waste",
            "discarded car tyre",
            "rubber waste material",
            "worn out tyre garbage",
        ],
    },
    {
        "name":        "Wood / Furniture",
        "key":         "wood",
        "icon":        "🪵",
        "color":       "#92400E",
        "impact":      3,
        "recyclable":  True,
        "hazardous":   False,
        "carbon_kg":   0.8,
        "value_usd":   0.20,
        "decompose":   "13 years",
        "prompts": [
            "wood waste timber scraps",
            "broken furniture wood",
            "wooden plank scrap waste",
            "sawdust wood offcuts waste",
        ],
    },
    {
        "name":        "Construction / Debris",
        "key":         "construction",
        "icon":        "🧱",
        "color":       "#B45309",
        "impact":      6,
        "recyclable":  True,
        "hazardous":   False,
        "carbon_kg":   2.1,
        "value_usd":   0.10,
        "decompose":   "500–1000 years",
        "prompts": [
            "construction debris waste",
            "broken brick concrete rubble",
            "demolition waste material",
            "building rubble waste",
        ],
    },
]

CAT_BY_KEY = {c["key"]: c for c in CATEGORIES}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _gen(prompt: str, max_new: int = 140) -> str:
    """Run a single Flan-T5 generation."""
    inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=160)
    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=max_new,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    return {
        "total": 0,
        "categories": {},
        "history": [],          # list of dicts: {time, name, icon, conf}
        "total_carbon_saved": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION  (ensemble prompts + text blend)
# ─────────────────────────────────────────────────────────────────────────────
def classify_waste(image, text: str):
    all_prompts, prompt_owner = [], []
    for i, cat in enumerate(CATEGORIES):
        for p in cat["prompts"]:
            all_prompts.append(p)
            prompt_owner.append(i)

    img_scores = np.zeros(len(CATEGORIES))
    if image is not None:
        inputs = clip_processor(
            text=all_prompts, images=image,
            return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            logits = clip_model(**inputs).logits_per_image[0].cpu().numpy()
        exp = np.exp(logits - logits.max()); probs = exp / exp.sum()
        for pi, ci in enumerate(prompt_owner):
            img_scores[ci] += probs[pi]
        img_scores /= img_scores.sum() + 1e-9

    txt_scores = np.zeros(len(CATEGORIES))
    if text.strip():
        u_enc = clip_processor(text=[text.strip()], return_tensors="pt", padding=True).to(device)
        c_enc = clip_processor(text=all_prompts,    return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            u = clip_model.get_text_features(**u_enc)
            c = clip_model.get_text_features(**c_enc)
        u = u / u.norm(dim=-1, keepdim=True)
        c = c / c.norm(dim=-1, keepdim=True)
        sims = (u @ c.T)[0].cpu().numpy()
        exp = np.exp(sims - sims.max()); probs = exp / exp.sum()
        for pi, ci in enumerate(prompt_owner):
            txt_scores[ci] += probs[pi]
        txt_scores /= txt_scores.sum() + 1e-9

    if image is not None and text.strip():
        final = 0.70 * img_scores + 0.30 * txt_scores
    elif image is not None:
        final = img_scores
    else:
        final = txt_scores

    top_idx = np.argsort(final)[::-1]
    best    = CATEGORIES[top_idx[0]]
    top5    = [(CATEGORIES[i], float(final[i])) for i in top_idx[:5]]
    return best, float(final[top_idx[0]]), top5


# ─────────────────────────────────────────────────────────────────────────────
# ENRICHED AI INSIGHT  –  9 distinct sections
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_insights(cat: dict) -> dict:
    name = cat["name"]
    return {
        "advantages": _gen(
            f"List 3 unique material advantages and positive properties of "
            f"{name} as a recyclable material. Be specific."
        ),
        "disadvantages": _gen(
            f"List 3 serious environmental and health risks caused by "
            f"improperly discarding {name}. Be concise."
        ),
        "renewal_industrial": _gen(
            f"List 4 industrial recycling and renewal methods for {name} waste. "
            f"Include processes like melting, shredding, chemical treatment."
        ),
        "renewal_diy": _gen(
            f"List 4 creative do-it-yourself (DIY) upcycling ideas for {name} waste "
            f"that anyone can do at home. Be practical and imaginative."
        ),
        "new_products": _gen(
            f"List 5 valuable new commercial products that can be manufactured "
            f"from recycled {name} waste. Be specific and inventive."
        ),
        "disposal_tips": _gen(
            f"Give 4 step-by-step safe disposal and preparation tips for "
            f"{name} waste before recycling. Include cleaning and sorting steps."
        ),
        "economic_value": _gen(
            f"Explain the economic value and financial incentives of recycling "
            f"{name} waste. Include job creation, resource savings and market value."
        ),
        "fun_facts": _gen(
            f"Share 3 surprising and interesting environmental facts about "
            f"{name} waste and recycling. Make them memorable."
        ),
        "global_stats": _gen(
            f"Share 3 important global statistics or data points about "
            f"{name} waste generation and recycling rates worldwide."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS  –  dual panel: donut + bar
# ─────────────────────────────────────────────────────────────────────────────
def plot_charts(state: dict):
    cats   = state["categories"]
    fig    = plt.figure(figsize=(10, 4), facecolor="#0f172a")
    spec   = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(spec[0])   # donut
    ax2 = fig.add_subplot(spec[1])   # horizontal bar

    for ax in (ax1, ax2):
        ax.set_facecolor("#0f172a")

    if not cats:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No Scans Yet", ha="center", va="center",
                    color="#334155", fontsize=11, transform=ax.transAxes)
            ax.axis("off")
        return fig

    labels  = [k.split("|")[0] for k in cats.keys()]
    values  = list(cats.values())
    colors  = [CAT_BY_KEY.get(k.split("|")[0], {}).get("color", "#22C55E")
               for k in cats.keys()]

    # ── Donut ─────────────────────────────────────────────────────────
    wedges, _, autotexts = ax1.pie(
        values, autopct="%1.0f%%", colors=colors,
        pctdistance=0.80,
        wedgeprops=dict(width=0.52, edgecolor="#0f172a", linewidth=2),
        textprops=dict(color="white", fontsize=7.5),
        startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(7.5); at.set_color("#F8FAFC")
    ax1.set_title(f"📊 Scan Distribution\n(Total: {state['total']})",
                  color="white", fontsize=9, fontweight="bold", pad=8)

    legend_patches = [mpatches.Patch(color=colors[i], label=labels[i])
                      for i in range(len(labels))]
    ax1.legend(handles=legend_patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.22), ncol=2,
               fontsize=6, frameon=False, labelcolor="white")

    # ── Horizontal bar ────────────────────────────────────────────────
    sorted_pairs = sorted(zip(labels, values, colors), key=lambda x: x[1])
    s_labels, s_values, s_colors = zip(*sorted_pairs) if sorted_pairs else ([], [], [])

    bars = ax2.barh(list(s_labels), list(s_values), color=list(s_colors),
                    height=0.55, edgecolor="#0f172a")
    for bar, val in zip(bars, s_values):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                 str(int(val)), va="center", color="white", fontsize=7.5)

    ax2.set_facecolor("#0f172a")
    ax2.tick_params(colors="white", labelsize=7.5)
    ax2.spines[:].set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.set_title("🔢 Count per Category",
                  color="white", fontsize=9, fontweight="bold", pad=8)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HTML BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def _conf_bar(c_name, c_icon, c_color, c_conf):
    pct = round(c_conf * 100, 1)
    return f"""
    <div style="margin-bottom:8px">
        <div style="display:flex;justify-content:space-between;
                    font-size:11px;color:#94A3B8;margin-bottom:3px">
            <span>{c_icon} {c_name}</span>
            <span style="color:{c_color};font-weight:700">{pct}%</span>
        </div>
        <div style="background:#1e293b;border-radius:99px;height:7px">
            <div style="background:{c_color};width:{pct}%;height:7px;
                        border-radius:99px"></div>
        </div>
    </div>"""


def _card(emoji, title, color, body_text, bullet_sep=","):
    """Generic card with bullet list parsed from body_text."""
    raw_items = [line.strip() for line in body_text.replace(";", ",").split(bullet_sep) if line.strip()]
    bullets   = "".join(
        f"<li style='margin-bottom:5px;line-height:1.6'>{item.lstrip('-•123456789. ')}</li>"
        for item in raw_items if item
    )
    return f"""
    <div style="
        background:#1e293b;
        border-left:4px solid {color};
        border-radius:0 12px 12px 0;
        padding:14px 16px;
        margin-bottom:10px;
    ">
        <div style="color:{color};font-weight:800;font-size:13px;
                    margin-bottom:8px;letter-spacing:.3px">{emoji} {title}</div>
        <ul style="color:#CBD5E1;font-size:12px;margin:0;
                   padding-left:18px;line-height:1.7">
            {bullets if bullets else f'<li>{body_text}</li>'}
        </ul>
    </div>"""


def _stat_pill(label, value, color="#22C55E"):
    return f"""
    <div style="
        background:#1e293b;
        border:1px solid {color}33;
        border-radius:12px;
        padding:10px 14px;
        text-align:center;
        flex:1;min-width:100px;
    ">
        <div style="color:{color};font-size:18px;font-weight:800">{value}</div>
        <div style="color:#64748B;font-size:10px;margin-top:2px;
                    letter-spacing:.5px;text-transform:uppercase">{label}</div>
    </div>"""


def _impact_meter(score: int, color: str):
    """Horizontal 10-block impact meter."""
    blocks = ""
    for i in range(1, 11):
        fill = color if i <= score else "#1e293b"
        blocks += f"""<div style="
            width:22px;height:12px;border-radius:3px;
            background:{fill};display:inline-block;margin-right:2px;
        "></div>"""
    label = (
        "🟢 LOW"    if score <= 3 else
        "🟡 MEDIUM" if score <= 6 else
        "🔴 HIGH"
    )
    return f"""
    <div style="margin-bottom:14px">
        <div style="font-size:11px;color:#64748B;margin-bottom:6px;
                    letter-spacing:1px;text-transform:uppercase">
            🌡️ Environmental Impact Score &nbsp;
            <span style="color:{color};font-weight:700">{score}/10 — {label}</span>
        </div>
        <div>{blocks}</div>
    </div>"""


def build_result_html(cat: dict, conf: float, top5: list, insights: dict) -> str:
    recyclable  = cat["recyclable"]
    hazardous   = cat["hazardous"]

    badge = (
        '<span style="background:#22C55E22;color:#22C55E;padding:3px 12px;'
        'border-radius:20px;font-weight:700;font-size:11px">✅ Recyclable</span>'
        if recyclable else
        '<span style="background:#EF444422;color:#EF4444;padding:3px 12px;'
        'border-radius:20px;font-weight:700;font-size:11px">⛔ Special Disposal Required</span>'
    )
    haz_badge = (
        '<span style="background:#F9731622;color:#F97316;padding:3px 10px;'
        'border-radius:20px;font-weight:700;font-size:11px">⚠️ Hazardous</span>'
        if hazardous else ""
    )

    top5_bars = "".join(
        _conf_bar(c["name"], c["icon"], c["color"], cf)
        for c, cf in top5
    )

    stats_row = f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px">
        {_stat_pill("Decompose Time", cat["decompose"], cat["color"])}
        {_stat_pill("Impact Score",   f"{cat['impact']}/10", "#EF4444")}
        {_stat_pill("Scrap Value",    f"~${cat['value_usd']}/kg", "#22C55E")}
        {_stat_pill("CO₂ if Landfill", f"~{cat['carbon_kg']} kg/kg", "#F97316")}
    </div>"""

    # ── Renewal tab-style sections ──
    industrial = _card("🏭", "Industrial Recycling Methods",   "#3B82F6", insights["renewal_industrial"])
    diy        = _card("🔨", "DIY Upcycling Ideas at Home",    "#F59E0B", insights["renewal_diy"])
    products   = _card("🛒", "New Products Made from This Waste", "#10B981", insights["new_products"])
    advantages = _card("✅", "Advantages of This Waste Type",  "#22C55E", insights["advantages"])
    disadv     = _card("⚠️", "Environmental Risks if Mishandled", "#EF4444", insights["disadvantages"])
    disposal   = _card("🚮", "Safe Disposal & Preparation Tips",  "#8B5CF6", insights["disposal_tips"])
    economic   = _card("💰", "Economic Value & Financial Impact", "#F59E0B", insights["economic_value"])
    fun_facts  = _card("🌍", "Surprising Environmental Facts",  "#EC4899", insights["fun_facts"])
    stats_txt  = _card("📊", "Global Waste Statistics",         "#64748B", insights["global_stats"])

    return f"""
    <div style="
        background:linear-gradient(160deg,#0f172a 60%,#162032);
        border:1px solid #1e293b;
        padding:22px;
        border-radius:20px;
        color:white;
        font-family:'Segoe UI',sans-serif;
    ">
        <!-- ── Header ── -->
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:16px">
            <div style="
                font-size:48px;background:#1e293b;width:76px;height:76px;
                border-radius:18px;display:flex;align-items:center;justify-content:center;
                border:2px solid {cat['color']}44;
            ">{cat['icon']}</div>
            <div>
                <div style="font-size:11px;color:#64748B;letter-spacing:2px;
                            text-transform:uppercase">Waste Detected</div>
                <div style="font-size:22px;font-weight:800;color:{cat['color']};
                            line-height:1.2">{cat['name']}</div>
                <div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap">
                    {badge}{haz_badge}
                </div>
            </div>
        </div>

        <!-- ── Impact Meter ── -->
        {_impact_meter(cat['impact'], cat['color'])}

        <!-- ── Quick Stats ── -->
        {stats_row}

        <!-- ── Confidence Bars ── -->
        <div style="background:#0f172a;border-radius:12px;padding:14px;margin-bottom:14px">
            <div style="font-size:11px;color:#64748B;margin-bottom:10px;
                        letter-spacing:1px;text-transform:uppercase">🎯 Top-5 Detection Confidence</div>
            {top5_bars}
        </div>

        <!-- ── Section Grid: 2 cols ── -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 12px">
            <div>{advantages}{industrial}{diy}</div>
            <div>{disadv}{disposal}{economic}</div>
        </div>

        <!-- ── Full-width sections ── -->
        {products}
        {fun_facts}
        {stats_txt}

        <div style="font-size:10px;color:#334155;margin-top:14px;text-align:right">
            EcoMind AI PRO+ · CLIP ViT-L/14 + Flan-T5-Large · {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
    """


def build_history_html(history: list) -> str:
    if not history:
        return "<p style='color:#334155;text-align:center;padding:20px'>No scan history yet.</p>"
    rows = "".join(
        f"""<tr>
            <td style="padding:8px 10px;color:#64748B;font-size:11px">{h['time']}</td>
            <td style="padding:8px 10px">{h['icon']} <span style="color:{h['color']};
                font-weight:600;font-size:12px">{h['name']}</span></td>
            <td style="padding:8px 10px;color:#22C55E;font-size:12px;
                font-weight:700">{h['conf']:.1f}%</td>
        </tr>"""
        for h in reversed(history[-20:])
    )
    return f"""
    <table style="width:100%;border-collapse:collapse;
                  font-family:'Segoe UI',sans-serif">
        <thead>
            <tr style="border-bottom:1px solid #1e293b">
                <th style="padding:8px 10px;color:#64748B;font-size:10px;
                           text-align:left;font-weight:600">TIME</th>
                <th style="padding:8px 10px;color:#64748B;font-size:10px;
                           text-align:left;font-weight:600">CATEGORY</th>
                <th style="padding:8px 10px;color:#64748B;font-size:10px;
                           text-align:left;font-weight:600">CONFIDENCE</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HANDLER
# ─────────────────────────────────────────────────────────────────────────────
def analyse(image, text, state):
    if image is None and not text.strip():
        return (
            "⚠️ Please upload an image or describe the waste.",
            "<p style='color:#94A3B8;padding:20px'>No input provided.</p>",
            None,
            build_history_html(state["history"]),
            state,
        )

    cat, conf, top5 = classify_waste(image, text)
    insights        = generate_all_insights(cat)

    # Update state
    state["total"] += 1
    key = cat["key"] + "|" + cat["icon"]
    state["categories"][key] = state["categories"].get(key, 0) + 1
    state["history"].append({
        "time":  datetime.datetime.now().strftime("%H:%M:%S"),
        "name":  cat["name"],
        "icon":  cat["icon"],
        "color": cat["color"],
        "conf":  conf * 100,
    })
    state["total_carbon_saved"] += cat["carbon_kg"] * 0.3   # rough offset estimate

    html        = build_result_html(cat, conf, top5, insights)
    status_msg  = (
        f"✅ {cat['icon']} {cat['name']}  ·  {conf*100:.1f}% confidence  ·  "
        f"Scan #{state['total']}  ·  "
        f"Est. CO₂ offset: ~{state['total_carbon_saved']:.1f} kg"
    )

    return (
        status_msg,
        html,
        plot_charts(state),
        build_history_html(state["history"]),
        state,
    )


def clear_fn(state):
    fresh = init_state()
    placeholder = """
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:20px;
                padding:30px;text-align:center;color:#334155;">
        <div style="font-size:42px;margin-bottom:10px">🌍</div>
        <p style="font-size:14px">Ready for next scan</p>
    </div>"""
    return None, "", placeholder, build_history_html([]), fresh


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --eco-green:  #22C55E;
    --eco-blue:   #3B82F6;
    --eco-amber:  #F59E0B;
    --eco-red:    #EF4444;
    --bg-base:    #0a0f1a;
    --bg-card:    #0f172a;
    --bg-lift:    #1e293b;
    --border:     #1e293b;
    --text-main:  #F8FAFC;
    --text-muted: #64748B;
}

html, body, .gradio-container {
    background: var(--bg-base) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-main) !important;
}

.gradio-container {
    max-width: 1300px !important;
    margin: auto !important;
    padding: 20px !important;
}

.svelte-1osuji4, .block {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}

label {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

input, textarea {
    background: var(--bg-lift) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

#analyse-btn {
    background: linear-gradient(135deg, #22C55E, #16A34A) !important;
    color: white !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 12px !important;
    height: 54px !important;
    letter-spacing: 0.5px !important;
    transition: transform .15s, box-shadow .15s !important;
}

#analyse-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px #22C55E44 !important;
}

#clear-btn {
    background: var(--bg-lift) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    height: 44px !important;
}

#clear-btn:hover {
    color: var(--eco-red) !important;
    border-color: var(--eco-red) !important;
}

#status-box textarea {
    background: var(--bg-lift) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: var(--eco-green) !important;
}


/* Detailed response text white */
ul, li, p {
    color: #FFFFFF !important;
}

/* Headings visible on dark background */
div[style*="font-weight:800"],
div[style*="font-weight:700"] {
    color: #64748B !important;
}

::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: var(--bg-base);
}

::-webkit-scrollbar-thumb {
    background: var(--bg-lift);
    border-radius: 99px;
}
"""
HERO = """
<div style="
    text-align:center;padding:34px 20px;
    background:linear-gradient(160deg,#0d1b0f 0%,#0a1628 50%,#0d0f1a 100%);
    border:1px solid #1e2f3a;border-radius:24px;margin-bottom:18px;
    position:relative;overflow:hidden;
">
    <div style="position:absolute;top:-40px;left:50%;transform:translateX(-50%);
                width:340px;height:130px;
                background:radial-gradient(ellipse,#22C55E33,transparent 70%);
                pointer-events:none"></div>

    <div style="font-size:52px;margin-bottom:8px;position:relative">🌍♻️🤖</div>

    <h1 style="
        font-family:'Space Grotesk',sans-serif;
        font-size:clamp(22px,4vw,38px);font-weight:800;
        background:linear-gradient(90deg,#22C55E,#3B82F6,#F59E0B);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        margin:0 0 8px;letter-spacing:-1px;
    ">EcoMind AI PRO+</h1>

    <p style="color:#64748B;font-size:14px;margin:0 0 18px;letter-spacing:.5px">
        Waste Intelligence Platform · Vision + Generative AI · 12 Categories · 9 Insight Modules
    </p>

    <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap">
        <span style="background:#22C55E18;border:1px solid #22C55E44;color:#22C55E;
                     padding:4px 14px;border-radius:99px;font-size:11px;font-weight:600">✅ Advantages</span>
        <span style="background:#3B82F618;border:1px solid #3B82F644;color:#3B82F6;
                     padding:4px 14px;border-radius:99px;font-size:11px;font-weight:600">🏭 Industrial Renewal</span>
        <span style="background:#F59E0B18;border:1px solid #F59E0B44;color:#F59E0B;
                     padding:4px 14px;border-radius:99px;font-size:11px;font-weight:600">🔨 DIY Upcycling</span>
        <span style="background:#10B98118;border:1px solid #10B98144;color:#10B981;
                     padding:4px 14px;border-radius:99px;font-size:11px;font-weight:600">🛒 New Products</span>
        <span style="background:#EF444418;border:1px solid #EF444444;color:#EF4444;
                     padding:4px 14px;border-radius:99px;font-size:11px;font-weight:600">⚠️ Risk Alerts</span>
        <span style="background:#8B5CF618;border:1px solid #8B5CF644;color:#8B5CF6;
                     padding:4px 14px;border-radius:99px;font-size:11px;font-weight:600">🚮 Disposal Tips</span>
        <span style="background:#EC489918;border:1px solid #EC489944;color:#EC4899;
                     padding:4px 14px;border-radius:99px;font-size:11px;font-weight:600">🌍 Fun Facts</span>
        <span style="background:#F9731618;border:1px solid #F9731644;color:#F97316;
                     padding:4px 14px;border-radius:99px;font-size:11px;font-weight:600">💰 Economic Value</span>
        <span style="background:#64748B18;border:1px solid #64748B44;color:#64748B;
                     padding:4px 14px;border-radius:99px;font-size:11px;font-weight:600">📊 Global Stats</span>
    </div>
</div>
"""

SUPPORTED = """
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
            padding:12px 16px;margin-top:8px;font-size:11px;color:#475569;
            display:flex;flex-wrap:wrap;gap:6px;align-items:center">
    <span style="color:#64748B;font-weight:600">12 Waste Types:</span>
    ♻️ Plastic &nbsp;·&nbsp; ⚙️ Metal &nbsp;·&nbsp; 📄 Paper &nbsp;·&nbsp;
    🔷 Glass &nbsp;·&nbsp; 🌱 Organic &nbsp;·&nbsp; ⚡ E-Waste &nbsp;·&nbsp;
    👕 Textile &nbsp;·&nbsp; ☢️ Hazardous &nbsp;·&nbsp; 🔋 Battery &nbsp;·&nbsp;
    🛞 Rubber &nbsp;·&nbsp; 🪵 Wood &nbsp;·&nbsp; 🧱 Construction
</div>
"""

PLACEHOLDER_HTML = """
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:20px;
            padding:40px;text-align:center;color:#334155;">
    <div style="font-size:52px;margin-bottom:12px">🌍</div>
    <p style="font-size:15px;margin:0 0 6px;color:#475569">Upload a waste image or describe what you have</p>
    <p style="font-size:12px;color:#334155">Get a full 9-module AI environmental intelligence report</p>
</div>"""

# ─────────────────────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="EcoMind AI PRO+", theme=gr.themes.Base()) as app:

    state = gr.State(init_state())
    gr.HTML(HERO)

    with gr.Row():
        # ── LEFT PANEL ────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            img = gr.Image(type="pil", label="Upload Waste Image", height=260)
            txt = gr.Textbox(
                label="Describe the Waste (optional — boosts accuracy)",
                placeholder="e.g. 'crushed aluminium cans', 'broken phone screen', 'old rubber tyre' …",
                lines=2,
            )
            btn   = gr.Button("🚀  Analyse Waste", elem_id="analyse-btn")
            clear = gr.Button("🗑️  Clear",          elem_id="clear-btn")

            out_text = gr.Textbox(
                label="Detection Result", elem_id="status-box",
                interactive=False, lines=2,
            )

            gr.HTML(SUPPORTED)

            # History panel
            gr.Markdown("### 📋 Scan History", elem_classes=[])
            history_html = gr.HTML(
                "<p style='color:#334155;text-align:center;padding:16px;font-size:12px'>No scans yet</p>"
            )

        # ── RIGHT PANEL ───────────────────────────────────────────────
        with gr.Column(scale=2):
            out_html = gr.HTML(PLACEHOLDER_HTML)
            chart    = gr.Plot(label="Session Analytics")

    # ── Callbacks ─────────────────────────────────────────────────────
    btn.click(
        fn=analyse,
        inputs=[img, txt, state],
        outputs=[out_text, out_html, chart, history_html, state],
    )

    clear.click(
        fn=clear_fn,
        inputs=[state],
        outputs=[img, txt, out_html, history_html, state],
    )

app.launch()