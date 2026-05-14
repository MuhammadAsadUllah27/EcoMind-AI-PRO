# ═════════════════════════════════════════════════════════════════════════════
#  EcoMind AI PRO+  –  Waste Classification & Environmental Intelligence App
#  Uses CLIP (vision model) + Flan-T5 (text generation) + Gradio (web UI)
# ═════════════════════════════════════════════════════════════════════════════

# ── Standard & third-party imports ───────────────────────────────────────────
import gradio as gr                          # Gradio: builds the interactive web UI
import torch                                 # PyTorch: deep learning framework
from transformers import (
    CLIPProcessor, CLIPModel,                # CLIP: image+text zero-shot model from OpenAI
    AutoTokenizer,                           # Automatically loads the right tokenizer for a model
    AutoModelForSeq2SeqLM,                   # Loads sequence-to-sequence (encoder-decoder) models
)
from PIL import Image                        # Pillow: image loading and manipulation
import matplotlib                            # Matplotlib: plotting library
matplotlib.use("Agg")                        # Use non-interactive backend (renders to file/buffer, not screen)
import matplotlib.pyplot as plt              # High-level plotting API
import matplotlib.patches as mpatches       # Used to create coloured legend boxes in charts
import matplotlib.gridspec as gridspec       # Fine-grained control over subplot layout
import numpy as np                           # NumPy: fast numerical arrays & math
import datetime                              # Standard library: timestamps for scan history


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

# If an NVIDIA GPU is available, use "cuda"; otherwise fall back to "cpu"
# Running on GPU is ~10-50× faster for large models like CLIP ViT-L
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[EcoMind] Using device: {device}")   # Log which device was chosen at startup


# ─────────────────────────────────────────────────────────────────────────────
# CLIP MODEL  –  Used for image & text classification (zero-shot)
# ─────────────────────────────────────────────────────────────────────────────

CLIP_MODEL_ID  = "openai/clip-vit-large-patch14"   # Hugging Face model identifier (ViT-L = best accuracy)
clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)     # Load CLIP weights, move to GPU/CPU
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)            # Tokenizer + image pre-processor for CLIP


# ─────────────────────────────────────────────────────────────────────────────
# GENERATIVE MODEL  –  Flan-T5 Large (encoder-decoder, instruction-tuned)
# ─────────────────────────────────────────────────────────────────────────────

GEN_MODEL_ID = "google/flan-t5-large"                                     # Hugging Face ID for Flan-T5 Large
tokenizer    = AutoTokenizer.from_pretrained(GEN_MODEL_ID)                # Loads T5's SentencePiece tokenizer
gen_model    = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_ID)        # Loads Flan-T5 weights (kept on CPU to save VRAM)


# ─────────────────────────────────────────────────────────────────────────────
# WASTE CATEGORIES  –  12 waste types with metadata for classification & display
# ─────────────────────────────────────────────────────────────────────────────

# Each category is a dict with:
#   name        – Human-readable label shown in the UI
#   key         – Short identifier used internally (state dict keys, lookup)
#   icon        – Emoji displayed in UI cards and history
#   color       – Hex colour for charts, borders, badges
#   impact      – Environmental impact score 1–10 (10 = most harmful if not recycled)
#   recyclable  – True if standard recycling facilities can handle this
#   hazardous   – True if special handling / disposal is legally required
#   carbon_kg   – kg of CO₂ emitted per kg if sent to landfill (approx.)
#   value_usd   – Approximate scrap market value in USD per kg
#   decompose   – How long this material takes to break down naturally
#   prompts     – Short text descriptions fed to CLIP for zero-shot matching

CATEGORIES = [
    {
        "name":        "Plastic Bottle / Container",
        "key":         "plastic",
        "icon":        "♻️",
        "color":       "#3B82F6",      # Blue
        "impact":      7,
        "recyclable":  True,
        "hazardous":   False,
        "carbon_kg":   1.9,
        "value_usd":   0.25,
        "decompose":   "450 years",
        # CLIP will compare the uploaded image against all of these text prompts
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
        "color":       "#6B7280",      # Grey
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
        "color":       "#F59E0B",      # Amber
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
        "color":       "#10B981",      # Emerald green
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
        "color":       "#84CC16",      # Lime green
        "impact":      4,
        "recyclable":  True,           # Can be composted
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
        "color":       "#8B5CF6",      # Violet
        "impact":      9,              # Very high – contains lead, mercury, cadmium
        "recyclable":  True,           # Specialised e-waste recyclers can recover metals
        "hazardous":   True,           # Contains toxic heavy metals
        "carbon_kg":   5.8,
        "value_usd":   2.00,           # Highest scrap value (precious metals inside)
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
        "color":       "#EC4899",      # Pink
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
        "color":       "#EF4444",      # Red – danger
        "impact":      10,             # Maximum impact – permanently toxic
        "recyclable":  False,          # Cannot be processed in normal facilities
        "hazardous":   True,
        "carbon_kg":   6.5,
        "value_usd":   0.00,           # No scrap value; costly to dispose safely
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
        "color":       "#F97316",      # Orange
        "impact":      9,
        "recyclable":  True,           # Battery recycling programmes exist
        "hazardous":   True,           # Lithium / acid / heavy metals inside
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
        "color":       "#78716C",      # Stone grey
        "impact":      7,
        "recyclable":  True,           # Can be ground into crumb rubber or pyrolysed
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
        "color":       "#92400E",      # Brown
        "impact":      3,              # Relatively low; biodegrades and can be chipped
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
        "color":       "#B45309",      # Dark amber/brown
        "impact":      6,
        "recyclable":  True,           # Concrete can be crushed and reused as aggregate
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

# Build a fast lookup dict:  "plastic" → {...category dict...}
# Used when we only have the short key and need the full metadata
CAT_BY_KEY = {c["key"]: c for c in CATEGORIES}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Run Flan-T5 inference to generate a text answer
# ─────────────────────────────────────────────────────────────────────────────

def _gen(prompt: str, max_new: int = 140) -> str:
    """
    Feed a text prompt to Flan-T5 and return its generated response.

    Args:
        prompt   : The instruction string (e.g. "List 3 recycling methods for …")
        max_new  : Maximum number of new tokens to generate (limits response length)

    Returns:
        Decoded string of the model's response, with special tokens removed.
    """
    # Tokenise the input string into model-readable tensors
    # truncation=True cuts prompts longer than 160 tokens to avoid errors
    inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=160)

    # Run beam-search decoding to generate the answer tokens
    # num_beams=4 means 4 parallel hypotheses are explored (better quality than greedy)
    # no_repeat_ngram_size=3 prevents the model repeating 3-gram phrases
    outputs = gen_model.generate(
        **inputs,
        max_new_tokens=max_new,
        num_beams=4,
        early_stopping=True,           # Stop as soon as all beams hit an EOS token
        no_repeat_ngram_size=3,
    )

    # Decode the token IDs back to a human-readable string; remove <pad>, </s>, etc.
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE  –  Tracks cumulative scan data across multiple uploads
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    """
    Create a fresh, empty session state dictionary.
    This is stored in Gradio's gr.State component so it persists across button clicks.

    Fields:
        total             – How many waste items have been scanned this session
        categories        – Dict of { "key|icon": count } for chart generation
        history           – List of recent scan records (time, name, confidence …)
        total_carbon_saved – Running estimate of CO₂ offset from proper recycling
    """
    return {
        "total": 0,
        "categories": {},
        "history": [],
        "total_carbon_saved": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION  –  CLIP-based image + text ensemble scorer
# ─────────────────────────────────────────────────────────────────────────────

def classify_waste(image, text: str):
    """
    Classify the waste item using CLIP's zero-shot image-text similarity.

    Strategy:
      1. Collect every prompt string from all 12 categories (4 prompts × 12 = 48 total).
      2. If an image was provided, compute image-vs-all-prompts similarity scores,
         then aggregate per category.
      3. If text was provided, compute text-vs-all-prompts similarity scores,
         then aggregate per category.
      4. Blend image scores (70%) + text scores (30%) if both are present.
      5. Return the top category and the top-5 ranked results.

    Args:
        image : PIL.Image or None
        text  : Optional user description string

    Returns:
        best  (dict)        – The winning category dict
        conf  (float)       – Winning category's blended confidence score (0-1)
        top5  (list)        – [(category_dict, score), …] for top 5 categories
    """

    # Build a flat list of all prompt strings and track which category each belongs to
    all_prompts  = []   # e.g. ["a plastic bottle", "plastic packaging waste", "an aluminium can", ...]
    prompt_owner = []   # parallel list of category indices, e.g. [0, 0, 0, 0, 1, 1, 1, 1, ...]

    for i, cat in enumerate(CATEGORIES):
        for p in cat["prompts"]:
            all_prompts.append(p)
            prompt_owner.append(i)          # each prompt maps back to its parent category index

    # ── IMAGE SCORING ─────────────────────────────────────────────────────────
    img_scores = np.zeros(len(CATEGORIES))  # one score slot per category, starts at 0

    if image is not None:
        # Prepare image + all text prompts together for CLIP
        inputs = clip_processor(
            text=all_prompts, images=image,
            return_tensors="pt", padding=True   # pad shorter tokenised prompts to same length
        ).to(device)                            # move tensors to GPU/CPU

        with torch.no_grad():                   # disable gradient tracking (inference only, saves memory)
            # logits_per_image: shape [1, num_prompts] – raw similarity scores image vs each prompt
            logits = clip_model(**inputs).logits_per_image[0].cpu().numpy()

        # Softmax: convert raw logits → probabilities that sum to 1
        exp = np.exp(logits - logits.max())     # subtract max for numerical stability
        probs = exp / exp.sum()

        # Aggregate prompt-level probabilities up to category-level scores
        for pi, ci in enumerate(prompt_owner):
            img_scores[ci] += probs[pi]         # add each prompt's probability to its category

        img_scores /= img_scores.sum() + 1e-9  # re-normalise to sum=1 (1e-9 avoids zero division)

    # ── TEXT SCORING ──────────────────────────────────────────────────────────
    txt_scores = np.zeros(len(CATEGORIES))

    if text.strip():                            # only run if user provided a non-empty description
        # Encode the user's text description as a CLIP text embedding
        u_enc = clip_processor(text=[text.strip()], return_tensors="pt", padding=True).to(device)
        # Encode all category prompts
        c_enc = clip_processor(text=all_prompts,    return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            u = clip_model.get_text_features(**u_enc)   # shape [1, embed_dim]
            c = clip_model.get_text_features(**c_enc)   # shape [num_prompts, embed_dim]

        # L2-normalise embeddings so dot product equals cosine similarity
        u = u / u.norm(dim=-1, keepdim=True)
        c = c / c.norm(dim=-1, keepdim=True)

        # Compute cosine similarity: user text vs every category prompt
        sims = (u @ c.T)[0].cpu().numpy()       # shape [num_prompts]

        # Softmax to turn similarities into probabilities
        exp = np.exp(sims - sims.max())
        probs = exp / exp.sum()

        # Aggregate to category scores (same pattern as image scoring)
        for pi, ci in enumerate(prompt_owner):
            txt_scores[ci] += probs[pi]
        txt_scores /= txt_scores.sum() + 1e-9

    # ── BLEND IMAGE + TEXT ────────────────────────────────────────────────────
    if image is not None and text.strip():
        final = 0.70 * img_scores + 0.30 * txt_scores   # weighted blend: image dominates
    elif image is not None:
        final = img_scores                               # image only
    else:
        final = txt_scores                               # text description only

    # Sort indices highest→lowest score, pick top-1 and top-5
    top_idx = np.argsort(final)[::-1]
    best    = CATEGORIES[top_idx[0]]                    # winning category dict
    top5    = [(CATEGORIES[i], float(final[i])) for i in top_idx[:5]]

    return best, float(final[top_idx[0]]), top5


# ─────────────────────────────────────────────────────────────────────────────
# AI INSIGHT GENERATION  –  9 distinct informational sections via Flan-T5
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_insights(cat: dict) -> dict:
    """
    For the detected waste category, use Flan-T5 to generate 9 text insight sections.
    Each section is produced by a purpose-written prompt asking about a specific angle.

    Args:
        cat : The winning category dictionary (contains cat["name"])

    Returns:
        A dict with 9 keys, each containing a generated multi-sentence string.
    """
    name = cat["name"]   # e.g. "Plastic Bottle / Container"

    return {
        # Material advantages and positive recyclable properties
        "advantages": _gen(
            f"List 3 unique material advantages and positive properties of "
            f"{name} as a recyclable material. Be specific."
        ),
        # Risks when not properly disposed of
        "disadvantages": _gen(
            f"List 3 serious environmental and health risks caused by "
            f"improperly discarding {name}. Be concise."
        ),
        # Industrial-scale recycling processes
        "renewal_industrial": _gen(
            f"List 4 industrial recycling and renewal methods for {name} waste. "
            f"Include processes like melting, shredding, chemical treatment."
        ),
        # Home DIY upcycling projects
        "renewal_diy": _gen(
            f"List 4 creative do-it-yourself (DIY) upcycling ideas for {name} waste "
            f"that anyone can do at home. Be practical and imaginative."
        ),
        # New products manufactured from the recycled material
        "new_products": _gen(
            f"List 5 valuable new commercial products that can be manufactured "
            f"from recycled {name} waste. Be specific and inventive."
        ),
        # Step-by-step tips for safe preparation before recycling
        "disposal_tips": _gen(
            f"Give 4 step-by-step safe disposal and preparation tips for "
            f"{name} waste before recycling. Include cleaning and sorting steps."
        ),
        # Economic and financial rationale for recycling this material
        "economic_value": _gen(
            f"Explain the economic value and financial incentives of recycling "
            f"{name} waste. Include job creation, resource savings and market value."
        ),
        # Memorable / surprising environmental facts
        "fun_facts": _gen(
            f"Share 3 surprising and interesting environmental facts about "
            f"{name} waste and recycling. Make them memorable."
        ),
        # Global scale – statistics on generation and recycling rates
        "global_stats": _gen(
            f"Share 3 important global statistics or data points about "
            f"{name} waste generation and recycling rates worldwide."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS  –  Donut chart + Horizontal bar chart rendered with Matplotlib
# ─────────────────────────────────────────────────────────────────────────────

def plot_charts(state: dict):
    """
    Render a dark-themed dual-panel figure showing session analytics.
      Left panel  – Donut chart of waste category proportions
      Right panel – Horizontal bar chart of counts per category

    Args:
        state : Current session state dict (reads 'categories' and 'total')

    Returns:
        matplotlib Figure object (Gradio displays this in a gr.Plot component)
    """
    cats = state["categories"]    # e.g. {"plastic|♻️": 3, "metal|⚙️": 1}

    # Create a 10×4-inch figure with dark background, using GridSpec for 2 panels
    fig  = plt.figure(figsize=(10, 4), facecolor="#0f172a")
    spec = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)   # 1 row, 2 columns, 35% gap

    ax1 = fig.add_subplot(spec[0])   # Left subplot → donut chart
    ax2 = fig.add_subplot(spec[1])   # Right subplot → bar chart

    # Apply dark background to both axes
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f172a")

    # If no scans yet, show placeholder text and exit early
    if not cats:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No Scans Yet", ha="center", va="center",
                    color="#334155", fontsize=11, transform=ax.transAxes)
            ax.axis("off")
        return fig

    # Extract lists of labels, values, and corresponding hex colours
    labels = [k.split("|")[0] for k in cats.keys()]    # strip the icon from the composite key
    values = list(cats.values())
    colors = [
        CAT_BY_KEY.get(k.split("|")[0], {}).get("color", "#22C55E")
        for k in cats.keys()
    ]

    # ── Donut chart ────────────────────────────────────────────────────────
    wedges, _, autotexts = ax1.pie(
        values,
        autopct="%1.0f%%",                             # show integer percentage labels
        colors=colors,
        pctdistance=0.80,                              # place % labels 80% of the way to edge
        wedgeprops=dict(width=0.52,                    # ring thickness (1 = full pie, 0 = thin ring)
                        edgecolor="#0f172a",           # dark gap between wedges
                        linewidth=2),
        textprops=dict(color="white", fontsize=7.5),
        startangle=90,                                 # start from the top of the circle
    )
    # Style the percentage labels
    for at in autotexts:
        at.set_fontsize(7.5)
        at.set_color("#F8FAFC")

    ax1.set_title(
        f"📊 Scan Distribution\n(Total: {state['total']})",
        color="white", fontsize=9, fontweight="bold", pad=8
    )

    # Add a colour legend below the donut
    legend_patches = [
        mpatches.Patch(color=colors[i], label=labels[i])
        for i in range(len(labels))
    ]
    ax1.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),   # position below the chart
        ncol=2,                         # 2 columns in the legend
        fontsize=6, frameon=False, labelcolor="white"
    )

    # ── Horizontal bar chart ───────────────────────────────────────────────
    # Sort bars ascending so the longest bar is at the top
    sorted_pairs = sorted(zip(labels, values, colors), key=lambda x: x[1])
    s_labels, s_values, s_colors = zip(*sorted_pairs) if sorted_pairs else ([], [], [])

    bars = ax2.barh(
        list(s_labels), list(s_values),
        color=list(s_colors),
        height=0.55,             # bar thickness
        edgecolor="#0f172a"      # dark outline
    )

    # Add count labels to the right of each bar
    for bar, val in zip(bars, s_values):
        ax2.text(
            bar.get_width() + 0.05,                    # x: just past the bar end
            bar.get_y() + bar.get_height() / 2,        # y: vertically centred on bar
            str(int(val)),
            va="center", color="white", fontsize=7.5
        )

    # Style the bar chart axes
    ax2.set_facecolor("#0f172a")
    ax2.tick_params(colors="white", labelsize=7.5)     # white axis tick labels
    ax2.spines[:].set_visible(False)                   # remove all border lines
    ax2.xaxis.set_visible(False)                       # hide x-axis numbers (value shown as labels)
    ax2.set_title("🔢 Count per Category",
                  color="white", fontsize=9, fontweight="bold", pad=8)

    plt.tight_layout()   # auto-adjust padding so nothing overlaps
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HTML BUILDER HELPERS  –  Small functions that produce inline HTML strings
# ─────────────────────────────────────────────────────────────────────────────

def _conf_bar(c_name, c_icon, c_color, c_conf):
    """
    Render a single confidence bar row (label on left, percentage on right,
    coloured fill bar underneath).

    Args:
        c_name  : Category display name
        c_icon  : Category emoji
        c_color : Hex colour for the fill and percentage text
        c_conf  : Float between 0 and 1 (confidence score)
    """
    pct = round(c_conf * 100, 1)   # convert to percentage with 1 decimal place
    return f"""
    <div style="margin-bottom:8px">
        <div style="display:flex;justify-content:space-between;
                    font-size:11px;color:#94A3B8;margin-bottom:3px">
            <span>{c_icon} {c_name}</span>
            <span style="color:{c_color};font-weight:700">{pct}%</span>
        </div>
        <div style="background:#1e293b;border-radius:99px;height:7px">
            <!-- Fill width is proportional to pct value -->
            <div style="background:{c_color};width:{pct}%;height:7px;
                        border-radius:99px"></div>
        </div>
    </div>"""


def _card(emoji, title, color, body_text, bullet_sep=","):
    """
    Build a dark info card with a coloured left border and a bullet list inside.

    The body_text string is split on commas (or semicolons) to produce individual
    bullet points. This matches the comma-separated output that Flan-T5 often returns.

    Args:
        emoji     : Leading emoji shown in the card title
        title     : Card heading text
        color     : Left border / title colour
        body_text : Raw generated text; split into bullet items
        bullet_sep: Delimiter to split body_text into list items (default ",")
    """
    # Split on the separator, strip whitespace, remove blanks
    raw_items = [line.strip() for line in body_text.replace(";", ",").split(bullet_sep) if line.strip()]

    # Convert each item into an HTML <li>, stripping common leading characters (-, •, 1., 2. …)
    bullets = "".join(
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
            <!-- Fall back to raw text as a single item if splitting produced nothing -->
            {bullets if bullets else f'<li>{body_text}</li>'}
        </ul>
    </div>"""


def _stat_pill(label, value, color="#22C55E"):
    """
    Render a small rounded stat box (metric value on top, label below).
    Multiple pills are laid out in a flex row inside build_result_html.
    """
    return f"""
    <div style="
        background:#1e293b;
        border:1px solid {color}33;        <!-- 33 = 20% opacity hex suffix -->
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
    """
    Draw a 10-block horizontal impact meter (like a battery indicator).
    Filled blocks use the category colour; empty blocks use a dark shade.

    Args:
        score : Integer 1–10 (from category["impact"])
        color : Category hex colour for filled blocks
    """
    blocks = ""
    for i in range(1, 11):
        fill = color if i <= score else "#1e293b"   # fill up to score, rest dark
        blocks += f"""<div style="
            width:22px;height:12px;border-radius:3px;
            background:{fill};display:inline-block;margin-right:2px;
        "></div>"""

    # Descriptive label based on score range
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RESULT HTML BUILDER  –  Assembles the full right-panel report card
# ─────────────────────────────────────────────────────────────────────────────

def build_result_html(cat: dict, conf: float, top5: list, insights: dict) -> str:
    """
    Combine all visual components into one large HTML string that Gradio renders
    as a rich report card.

    Args:
        cat      : Winning category dict
        conf     : Winning score (0-1)
        top5     : List of (category, score) tuples for top-5 confidence bars
        insights : Dict of 9 generated text sections from generate_all_insights()
    """
    recyclable = cat["recyclable"]
    hazardous  = cat["hazardous"]

    # ── Recyclability badge ────────────────────────────────────────────────
    badge = (
        '<span style="background:#22C55E22;color:#22C55E;padding:3px 12px;'
        'border-radius:20px;font-weight:700;font-size:11px">✅ Recyclable</span>'
        if recyclable else
        '<span style="background:#EF444422;color:#EF4444;padding:3px 12px;'
        'border-radius:20px;font-weight:700;font-size:11px">⛔ Special Disposal Required</span>'
    )

    # ── Optional hazardous badge (only shown if cat is hazardous) ─────────
    haz_badge = (
        '<span style="background:#F9731622;color:#F97316;padding:3px 10px;'
        'border-radius:20px;font-weight:700;font-size:11px">⚠️ Hazardous</span>'
        if hazardous else ""
    )

    # ── Top-5 confidence progress bars ────────────────────────────────────
    top5_bars = "".join(
        _conf_bar(c["name"], c["icon"], c["color"], cf)
        for c, cf in top5
    )

    # ── 4 quick-fact pills in a flex row ──────────────────────────────────
    stats_row = f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px">
        {_stat_pill("Decompose Time", cat["decompose"], cat["color"])}
        {_stat_pill("Impact Score",   f"{cat['impact']}/10", "#EF4444")}
        {_stat_pill("Scrap Value",    f"~${cat['value_usd']}/kg", "#22C55E")}
        {_stat_pill("CO₂ if Landfill", f"~{cat['carbon_kg']} kg/kg", "#F97316")}
    </div>"""

    # ── Build each of the 9 AI insight cards ──────────────────────────────
    industrial = _card("🏭", "Industrial Recycling Methods",      "#3B82F6", insights["renewal_industrial"])
    diy        = _card("🔨", "DIY Upcycling Ideas at Home",       "#F59E0B", insights["renewal_diy"])
    products   = _card("🛒", "New Products Made from This Waste", "#10B981", insights["new_products"])
    advantages = _card("✅", "Advantages of This Waste Type",     "#22C55E", insights["advantages"])
    disadv     = _card("⚠️", "Environmental Risks if Mishandled","#EF4444", insights["disadvantages"])
    disposal   = _card("🚮", "Safe Disposal & Preparation Tips",  "#8B5CF6", insights["disposal_tips"])
    economic   = _card("💰", "Economic Value & Financial Impact", "#F59E0B", insights["economic_value"])
    fun_facts  = _card("🌍", "Surprising Environmental Facts",    "#EC4899", insights["fun_facts"])
    stats_txt  = _card("📊", "Global Waste Statistics",           "#64748B", insights["global_stats"])

    # ── Assemble the full HTML report card ────────────────────────────────
    return f"""
    <div style="
        background:linear-gradient(160deg,#0f172a 60%,#162032);
        border:1px solid #1e293b;
        padding:22px;
        border-radius:20px;
        color:white;
        font-family:'Segoe UI',sans-serif;
    ">
        <!-- ── Header: icon + name + recyclable/hazardous badges ── -->
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

        <!-- ── 10-block impact meter ── -->
        {_impact_meter(cat['impact'], cat['color'])}

        <!-- ── 4 quick-stat pills ── -->
        {stats_row}

        <!-- ── Top-5 CLIP confidence bars ── -->
        <div style="background:#0f172a;border-radius:12px;padding:14px;margin-bottom:14px">
            <div style="font-size:11px;color:#64748B;margin-bottom:10px;
                        letter-spacing:1px;text-transform:uppercase">🎯 Top-5 Detection Confidence</div>
            {top5_bars}
        </div>

        <!-- ── 2-column grid: left column = advantages / industrial / diy
                               right column = risks / disposal / economic ── -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0 12px">
            <div>{advantages}{industrial}{diy}</div>
            <div>{disadv}{disposal}{economic}</div>
        </div>

        <!-- ── Full-width sections below the grid ── -->
        {products}
        {fun_facts}
        {stats_txt}

        <!-- ── Tiny attribution footer ── -->
        <div style="font-size:10px;color:#334155;margin-top:14px;text-align:right">
            EcoMind AI PRO+ · CLIP ViT-L/14 + Flan-T5-Large · {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
    """


def build_history_html(history: list) -> str:
    """
    Render the scan history as an HTML table showing the last 20 scans
    in reverse-chronological order (most recent at the top).

    Args:
        history : List of scan record dicts [{time, name, icon, color, conf}, …]
    """
    if not history:
        return "<p style='color:#334155;text-align:center;padding:20px'>No scan history yet.</p>"

    # Build one <tr> per scan; reversed() shows newest first; [-20:] caps at last 20 entries
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
# MAIN EVENT HANDLER  –  Called when the user clicks "Analyse Waste"
# ─────────────────────────────────────────────────────────────────────────────

def analyse(image, text, state):
    """
    Orchestrates the full analysis pipeline:
      1. Validate that at least one input (image or text) was provided.
      2. Classify the waste item (CLIP ensemble).
      3. Generate all 9 AI insight sections (Flan-T5).
      4. Update session state (totals, history, CO₂ estimate).
      5. Build and return all UI outputs.

    Args:
        image : PIL.Image or None (from gr.Image component)
        text  : User description string (from gr.Textbox)
        state : Current session state dict (from gr.State)

    Returns:
        Tuple of 5 values that map to the 5 Gradio output components:
          (status_text, result_html, chart_figure, history_html, updated_state)
    """

    # Guard: if no input was given, return an early warning message
    if image is None and not text.strip():
        return (
            "⚠️ Please upload an image or describe the waste.",
            "<p style='color:#94A3B8;padding:20px'>No input provided.</p>",
            None,                                    # no chart update
            build_history_html(state["history"]),    # keep existing history
            state,                                   # state unchanged
        )

    # ── Step 1: Classify ──────────────────────────────────────────────────
    cat, conf, top5 = classify_waste(image, text)

    # ── Step 2: Generate insights ─────────────────────────────────────────
    insights = generate_all_insights(cat)

    # ── Step 3: Update session state ─────────────────────────────────────
    state["total"] += 1   # increment global scan counter

    # Use "key|icon" as the composite dict key so the chart can recover colour later
    key = cat["key"] + "|" + cat["icon"]
    state["categories"][key] = state["categories"].get(key, 0) + 1   # increment category counter

    # Append a scan record to the history list
    state["history"].append({
        "time":  datetime.datetime.now().strftime("%H:%M:%S"),
        "name":  cat["name"],
        "icon":  cat["icon"],
        "color": cat["color"],
        "conf":  conf * 100,    # store as percentage
    })

    # Rough CO₂ offset estimate: if properly recycled instead of landfilled,
    # approximately 30% of the landfill-emission figure is avoided
    state["total_carbon_saved"] += cat["carbon_kg"] * 0.3

    # ── Step 4: Build UI outputs ──────────────────────────────────────────
    html = build_result_html(cat, conf, top5, insights)

    # Status bar text shown in the small textbox above the result panel
    status_msg = (
        f"✅ {cat['icon']} {cat['name']}  ·  {conf*100:.1f}% confidence  ·  "
        f"Scan #{state['total']}  ·  "
        f"Est. CO₂ offset: ~{state['total_carbon_saved']:.1f} kg"
    )

    return (
        status_msg,
        html,
        plot_charts(state),                      # regenerate charts with new data
        build_history_html(state["history"]),
        state,                                   # return updated state back to gr.State
    )


def clear_fn(state):
    """
    Reset the entire session: clear image, text, result panel, history, and state.
    Called when the user clicks the "Clear" button.

    Args:
        state : Existing session state (discarded)

    Returns:
        Tuple of 5 reset values mapped to Gradio output components.
    """
    fresh = init_state()   # brand-new empty state

    # Placeholder HTML shown in the result panel before any scan
    placeholder = """
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:20px;
                padding:30px;text-align:center;color:#334155;">
        <div style="font-size:42px;margin-bottom:10px">🌍</div>
        <p style="font-size:14px">Ready for next scan</p>
    </div>"""

    return (
        None,                        # clear the image upload component
        "",                          # clear the text input
        placeholder,                 # reset result panel to placeholder
        build_history_html([]),      # empty history table
        fresh,                       # reset state
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS  –  Global dark-theme stylesheet injected into the Gradio page
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
/* Import Google Fonts: Space Grotesk (UI text) and JetBrains Mono (status output) */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* CSS custom properties (variables) for consistent theming */
:root {
    --eco-green:  #22C55E;
    --eco-blue:   #3B82F6;
    --eco-amber:  #F59E0B;
    --eco-red:    #EF4444;
    --bg-base:    #0a0f1a;   /* darkest background layer */
    --bg-card:    #0f172a;   /* card / panel background */
    --bg-lift:    #1e293b;   /* slightly elevated surface */
    --border:     #1e293b;
    --text-main:  #F8FAFC;
    --text-muted: #64748B;
}

/* Force dark background on the entire page and Gradio container */
html, body, .gradio-container {
    background: var(--bg-base) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text-main) !important;
}

/* Centre and constrain the app width */
.gradio-container {
    max-width: 1300px !important;
    margin: auto !important;
    padding: 20px !important;
}

/* Dark card background and border for Gradio block elements */
.svelte-1osuji4, .block {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
}

/* Style all form labels (uppercase, muted colour) */
label {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* Dark inputs and textareas */
input, textarea {
    background: var(--bg-lift) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Primary action button – green gradient with hover animation */
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
    transform: translateY(-2px) !important;              /* subtle lift on hover */
    box-shadow: 0 8px 28px #22C55E44 !important;         /* green glow */
}

/* Secondary clear button – dark with red hover state */
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

/* Status output box – monospace green text (like a terminal) */
#status-box textarea {
    background: var(--bg-lift) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: var(--eco-green) !important;
}

/* Ensure bullet list text is visible on dark backgrounds */
ul, li, p {
    color: #FFFFFF !important;
}

/* Ensure bold headings inside generated HTML are visible */
div[style*="font-weight:800"],
div[style*="font-weight:700"] {
    color: #64748B !important;
}

/* Custom scrollbar styling for dark theme */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--bg-lift); border-radius: 99px; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# STATIC HTML STRINGS  –  Hero banner and support info rendered once at load
# ─────────────────────────────────────────────────────────────────────────────

# Large gradient hero banner at the top of the app
HERO = """
<div style="
    text-align:center;padding:34px 20px;
    background:linear-gradient(160deg,#0d1b0f 0%,#0a1628 50%,#0d0f1a 100%);
    border:1px solid #1e2f3a;border-radius:24px;margin-bottom:18px;
    position:relative;overflow:hidden;
">
    <!-- Decorative radial glow behind the emoji row -->
    <div style="position:absolute;top:-40px;left:50%;transform:translateX(-50%);
                width:340px;height:130px;
                background:radial-gradient(ellipse,#22C55E33,transparent 70%);
                pointer-events:none"></div>

    <div style="font-size:52px;margin-bottom:8px;position:relative">🌍♻️🤖</div>

    <!-- App title with a 3-colour gradient text effect -->
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

    <!-- Row of feature tag pills describing each insight section -->
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

# Small info strip showing all 12 supported waste category icons
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

# Default placeholder shown in the result panel before the first scan
PLACEHOLDER_HTML = """
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:20px;
            padding:40px;text-align:center;color:#334155;">
    <div style="font-size:52px;margin-bottom:12px">🌍</div>
    <p style="font-size:15px;margin:0 0 6px;color:#475569">Upload a waste image or describe what you have</p>
    <p style="font-size:12px;color:#334155">Get a full 9-module AI environmental intelligence report</p>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI LAYOUT  –  Defines all components and wires callbacks
# ─────────────────────────────────────────────────────────────────────────────

# gr.Blocks() lets us arrange components in a custom layout (vs gr.Interface auto-layout)
# css= injects our dark-theme stylesheet; theme= sets the base Gradio theme
with gr.Blocks(css=CSS, title="EcoMind AI PRO+", theme=gr.themes.Base()) as app:

    # gr.State holds session-level data that persists across button clicks
    # (Gradio passes it as a function argument and returns the updated version)
    state = gr.State(init_state())

    gr.HTML(HERO)   # Render the hero banner HTML at the top of the page

    with gr.Row():   # Horizontal layout: left panel | right panel

        # ── LEFT PANEL: inputs, controls, status, history ─────────────────
        with gr.Column(scale=1, min_width=300):

            # Image upload widget; type="pil" means Gradio gives us a PIL.Image object
            img = gr.Image(type="pil", label="Upload Waste Image", height=260)

            # Optional text description to boost classification accuracy
            txt = gr.Textbox(
                label="Describe the Waste (optional — boosts accuracy)",
                placeholder="e.g. 'crushed aluminium cans', 'broken phone screen', 'old rubber tyre' …",
                lines=2,
            )

            # Primary CTA button – triggers the full analysis pipeline
            btn   = gr.Button("🚀  Analyse Waste", elem_id="analyse-btn")

            # Secondary button – resets everything back to a clean state
            clear = gr.Button("🗑️  Clear",          elem_id="clear-btn")

            # Read-only textbox showing classification result + session stats
            out_text = gr.Textbox(
                label="Detection Result", elem_id="status-box",
                interactive=False, lines=2,
            )

            gr.HTML(SUPPORTED)   # Render the 12-category icon strip

            # Section heading for the scan history table
            gr.Markdown("### 📋 Scan History", elem_classes=[])

            # HTML component to render the scan history table (updated after each scan)
            history_html = gr.HTML(
                "<p style='color:#334155;text-align:center;padding:16px;font-size:12px'>No scans yet</p>"
            )

        # ── RIGHT PANEL: result card + analytics chart ────────────────────
        with gr.Column(scale=2):   # scale=2 makes this panel twice as wide as the left

            # The main result card (rich HTML generated by build_result_html)
            out_html = gr.HTML(PLACEHOLDER_HTML)

            # Matplotlib figure displayed as a chart (donut + bar)
            chart = gr.Plot(label="Session Analytics")


    # ── BUTTON CALLBACKS ──────────────────────────────────────────────────

    # When Analyse is clicked:
    #   inputs  = [image component, text component, state component]
    #   outputs = [status textbox, result HTML, chart, history HTML, state]
    btn.click(
        fn=analyse,
        inputs=[img, txt, state],
        outputs=[out_text, out_html, chart, history_html, state],
    )

    # When Clear is clicked:
    #   inputs  = [state] (needed to read current state, though we discard it)
    #   outputs = [image, text, result HTML, history HTML, state]
    clear.click(
        fn=clear_fn,
        inputs=[state],
        outputs=[img, txt, out_html, history_html, state],
    )


# ── Launch the Gradio web server ──────────────────────────────────────────────
# By default this starts a local server at http://127.0.0.1:7860
# Pass share=True to get a temporary public URL via Gradio's tunnel
app.launch()