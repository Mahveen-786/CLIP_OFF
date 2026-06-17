import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

TEMPLATES = [
    "a photo of a {}",
    "a picture of a {}",
    "an image of a {}",
    "a photograph of a {}",
]

CLASS_EMOJIS = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦", "cat": "🐱",
    "deer": "🦌", "dog": "🐶", "frog": "🐸", "horse": "🐴",
    "ship": "🚢", "truck": "🚚",
}

MODEL_META = {
    "CLIP ViT-B/16": {"color": "#818cf8", "glow": "#818cf822", "org": "OpenAI",     "params": "150M", "zs": 89.9, "r1": 70.4},
    "BLIP-base":     {"color": "#34d399", "glow": "#34d39922", "org": "Salesforce", "params": "247M", "zs": 80.5, "r1": 86.8},
    "ALIGN-base":    {"color": "#fbbf24", "glow": "#fbbf2422", "org": "Google",     "params": "172M", "zs": 77.4, "r1": 81.6},
    "FLAVA-full":    {"color": "#f472b6", "glow": "#f472b622", "org": "Meta AI",    "params": "241M", "zs": 87.8, "r1": 68.8},
}

_models = {}

def load_clip():
    if "clip" not in _models:
        from transformers import CLIPProcessor, CLIPModel
        proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).eval()
        _models["clip"] = (proc, model)
    return _models["clip"]

def load_blip():
    if "blip" not in _models:
        from transformers import BlipProcessor, BlipForImageTextRetrieval
        proc  = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
        model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco").to(DEVICE).eval()
        _models["blip"] = (proc, model)
    return _models["blip"]

def load_align():
    if "align" not in _models:
        from transformers import AlignProcessor, AlignModel
        proc  = AlignProcessor.from_pretrained("kakaobrain/align-base")
        model = AlignModel.from_pretrained("kakaobrain/align-base").to(DEVICE).eval()
        _models["align"] = (proc, model)
    return _models["align"]

def load_flava():
    if "flava" not in _models:
        from transformers import FlavaProcessor, FlavaModel
        proc  = FlavaProcessor.from_pretrained("facebook/flava-full")
        model = FlavaModel.from_pretrained("facebook/flava-full", torch_dtype=torch.float32).to(DEVICE).eval()
        _models["flava"] = (proc, model)
    return _models["flava"]

def ensemble_zs_embs(embed_fn, classes, templates):
    stacked = []
    for tmpl in templates:
        stacked.append(embed_fn([tmpl.format(c) for c in classes]))
    avg = np.stack(stacked).mean(axis=0)
    return avg / np.linalg.norm(avg, axis=1, keepdims=True)

@torch.no_grad()
def clip_predict(image):
    proc, model = load_clip()
    inp   = proc(images=image, return_tensors="pt").to(DEVICE)
    res   = model.vision_model(pixel_values=inp["pixel_values"])
    i_emb = F.normalize(model.visual_projection(res.pooler_output).float(), dim=-1).cpu().numpy()
    def et(texts):
        i2 = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        r2 = model.text_model(input_ids=i2["input_ids"], attention_mask=i2["attention_mask"])
        return F.normalize(model.text_projection(r2.pooler_output).float(), dim=-1).cpu().numpy()
    return (i_emb @ ensemble_zs_embs(et, CLASSES, TEMPLATES).T)[0]

@torch.no_grad()
def blip_predict(image):
    proc, model = load_blip()
    inp   = proc(images=image, return_tensors="pt").to(DEVICE)
    vis   = model.vision_model(pixel_values=inp["pixel_values"])
    i_emb = F.normalize(model.vision_proj(vis.last_hidden_state[:, 0, :]), dim=-1).cpu().numpy()
    def et(texts):
        i2   = proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=35).to(DEVICE)
        tout = model.text_encoder(input_ids=i2["input_ids"], attention_mask=i2["attention_mask"], return_dict=True)
        return F.normalize(model.text_proj(tout.last_hidden_state[:, 0, :]), dim=-1).cpu().numpy()
    return (i_emb @ ensemble_zs_embs(et, CLASSES, TEMPLATES).T)[0]

@torch.no_grad()
def align_predict(image):
    proc, model = load_align()
    inp   = proc(images=image, return_tensors="pt").to(DEVICE)
    i_emb = F.normalize(model.get_image_features(**inp).pooler_output.float(), dim=-1).cpu().numpy()
    def et(texts):
        i2 = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        return F.normalize(model.get_text_features(**i2).pooler_output.float(), dim=-1).cpu().numpy()
    return (i_emb @ ensemble_zs_embs(et, CLASSES, TEMPLATES).T)[0]

@torch.no_grad()
def flava_predict(image):
    proc, model = load_flava()
    def et(texts):
        i2   = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        embs = model.get_text_features(**i2).pooler_output
        if embs.dim() == 3: embs = embs[:, 0, :]
        return F.normalize(embs.float(), dim=-1).cpu().numpy()
    inp  = proc(images=image, return_tensors="pt").to(DEVICE)
    embs = model.get_image_features(**inp).pooler_output
    if embs.dim() == 3: embs = embs[:, 0, :]
    i_emb = F.normalize(embs.float(), dim=-1).cpu().numpy()
    return (i_emb @ ensemble_zs_embs(et, CLASSES, TEMPLATES).T)[0]

PREDICT_FNS = {
    "CLIP ViT-B/16": clip_predict,
    "BLIP-base":     blip_predict,
    "ALIGN-base":    align_predict,
    "FLAVA-full":    flava_predict,
}

def to_probs(scores):
    e = np.exp(scores - scores.max())
    return e / e.sum()

def predict(image, selected_models):
    if image is None:
        return empty_state("Upload an image to begin the battle")
    if not selected_models:
        return empty_state("Select at least one model above")

    image = Image.fromarray(image).convert("RGB")
    cards = []

    # Keeping the processing order consistent: Row 1 (CLIP, BLIP) -> Row 2 (ALIGN, FLAVA)
    for name in ["CLIP ViT-B/16", "BLIP-base", "ALIGN-base", "FLAVA-full"]:
        if name not in selected_models:
            continue
        m     = MODEL_META[name]
        color = m["color"]
        t0    = time.time()
        try:
            raw_scores = PREDICT_FNS[name](image)
            scores     = raw_scores * 100.0  
            ms         = (time.time() - t0) * 1000
            probs      = to_probs(scores)
            top_idx    = int(np.argmax(probs))
            top_class  = CLASSES[top_idx]
            top_prob   = probs[top_idx] * 100
            ranking    = np.argsort(probs)[::-1]

            bars = ""
            for rank, idx in enumerate(ranking[:5]):
                cls  = CLASSES[idx]
                pct  = probs[idx] * 100
                bold = "font-weight:800;" if rank == 0 else "font-weight:400;"
                op   = "1" if rank == 0 else ("0.65" if rank < 3 else "0.35")
                bars += f"""
                <div style="display:flex;align-items:center;gap:8px;margin:6px 0;opacity:{op}">
                  <span style="width:105px;font-size:12px;{bold}color:#e2e8f0;white-space:nowrap">
                    {CLASS_EMOJIS[cls]} {cls}
                  </span>
                  <div style="flex:1;background:#1e293b;border-radius:99px;height:7px">
                    <div style="width:{pct:.1f}%;background:{color};height:7px;
                                border-radius:99px"></div>
                  </div>
                  <span style="width:40px;text-align:right;font-size:11px;{bold}color:{color}">
                    {pct:.1f}%
                  </span>
                </div>"""

            cards.append(f"""
            <div style="background:#0f172a;border:1px solid {color}55;border-radius:14px;
                        padding:18px;width:100%">
              <div style="display:flex;justify-content:space-between;align-items:center;
                          margin-bottom:14px">
                <div>
                  <div style="font-size:13px;font-weight:800;color:{color}">{name}</div>
                  <div style="font-size:10px;color:#475569;margin-top:2px">{m['org']} · {m['params']}</div>
                </div>
                <div style="background:{color}22;border:1px solid {color}44;color:{color};
                            border-radius:6px;padding:2px 8px;font-size:11px;font-weight:700">
                  {ms:.0f}ms
                </div>
              </div>
              <div style="background:#020617;border:1px solid {color}33;border-radius:10px;
                          padding:14px;text-align:center;margin-bottom:14px">
                <div style="font-size:36px">{CLASS_EMOJIS[top_class]}</div>
                <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                            letter-spacing:2px;margin:4px 0">Prediction</div>
                <div style="font-size:18px;font-weight:900;color:#f1f5f9;
                            text-transform:uppercase">{top_class}</div>
                <div style="font-size:28px;font-weight:900;color:{color}">{top_prob:.1f}%</div>
              </div>
              <div style="font-size:10px;color:#334155;text-transform:uppercase;
                          letter-spacing:1px;margin-bottom:8px;font-weight:700">Top 5</div>
              {bars}
              <div style="display:flex;justify-content:space-around;margin-top:14px;
                          padding-top:12px;border-top:1px solid #1e293b">
                <div style="text-align:center">
                  <div style="font-size:12px;font-weight:800;color:{color}">{m['zs']}%</div>
                  <div style="font-size:9px;color:#334155;text-transform:uppercase">CIFAR ZS</div>
                </div>
                <div style="text-align:center">
                  <div style="font-size:12px;font-weight:800;color:{color}">{m['r1']}%</div>
                  <div style="font-size:9px;color:#334155;text-transform:uppercase">Flickr R@1</div>
                </div>
              </div>
            </div>""")

        except Exception as e:
            cards.append(f"""
            <div style="background:#0f172a;border:1px solid #ef444455;border-radius:14px;
                        padding:18px;width:100%">
              <div style="color:#ef4444;font-weight:700;margin-bottom:6px">{name}</div>
              <div style="color:#f87171;font-size:12px">{str(e)}</div>
            </div>""")

    # Formats structural execution output boxes cleanly into a side-by-side 2-column layout
    return f"""
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:16px;width:100%">
      {"".join(cards)}
    </div>"""

def empty_state(msg):
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:300px;background:#0f172a;border-radius:14px;
                border:1px dashed #1e293b;width:100%">
      <div style="font-size:40px;margin-bottom:12px">🏟️</div>
      <div style="font-size:14px;font-weight:600;color:#334155">{msg}</div>
    </div>"""

PAGE_HTML = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
  html, body { background:#020617 !important; margin:0; padding:0; }
  * { font-family:'Inter',sans-serif !important; box-sizing:border-box; }
  .gradio-container { background:#020617 !important; max-width:1200px !important;
                      margin:auto !important; padding:0 !important; }
  .contain { background:#020617 !important; }
  footer { display:none !important; }
  .upload-container, [data-testid="image"] { background:#0f172a !important;
    border:1px dashed #334155 !important; border-radius:12px !important; }
  .wrap { background:#0f172a !important; border:1px solid #1e293b !important;
          border-radius:12px !important; padding:12px !important; }
  .wrap label span { color:#94a3b8 !important; font-size:13px !important; font-weight:600 !important; }
  button.primary { background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border:none !important; border-radius:10px !important; font-size:15px !important;
    font-weight:700 !important; color:white !important;
    box-shadow:0 0 24px #6366f144 !important; width:100% !important; padding:13px !important; }
  button.primary:hover { box-shadow:0 0 40px #6366f166 !important; }
  label.svelte-1b6s6s { display:none !important; }
</style>

<div style="background:linear-gradient(135deg,#0f172a 0%,#1a1040 50%,#0f172a 100%);
            border-bottom:1px solid #1e293b;padding:24px 32px;margin-bottom:20px;
            display:flex;align-items:center;gap:18px;flex-wrap:wrap">
  <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);border-radius:12px;
              width:52px;height:52px;display:flex;align-items:center;justify-content:center;
              font-size:26px;flex-shrink:0;box-shadow:0 0 28px #6366f155">✂️</div>
  <div style="flex:1">
    <div style="font-size:26px;font-weight:900;color:white;letter-spacing:-0.5px">
      CLIP OFF
    </div>
    <div style="font-size:12px;color:#475569;margin-top:3px;font-weight:500">
      Zero-Shot Vision-Language Model Benchmark &nbsp;·&nbsp;
      CLIP &nbsp;·&nbsp; BLIP &nbsp;·&nbsp; ALIGN &nbsp;·&nbsp; FLAVA
    </div>
  </div>
  <div style="display:flex;gap:20px">
    <div style="text-align:center">
      <div style="font-size:20px;font-weight:900;color:#818cf8">4</div>
      <div style="font-size:9px;color:#334155;text-transform:uppercase;letter-spacing:1px">Models</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:20px;font-weight:900;color:#34d399">10k</div>
      <div style="font-size:9px;color:#334155;text-transform:uppercase;letter-spacing:1px">Samples</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:20px;font-weight:900;color:#fbbf24">10</div>
      <div style="font-size:9px;color:#334155;text-transform:uppercase;letter-spacing:1px">Classes</div>
    </div>
    <div style="text-align:center">
      <div style="font-size:20px;font-weight:900;color:#f472b6">4×</div>
      <div style="font-size:9px;color:#334155;text-transform:uppercase;letter-spacing:1px">Prompts</div>
    </div>
  </div>
</div>

<div style="display:flex;gap:10px;flex-wrap:wrap;padding:0 24px;margin-bottom:20px">
  <div style="flex:1;min-width:140px;background:#0f172a;border:1px solid #818cf833;
              border-radius:12px;padding:14px 16px">
    <div style="font-size:9px;color:#334155;text-transform:uppercase;letter-spacing:1px;
                margin-bottom:4px">Best Zero-Shot</div>
    <div style="font-size:22px;font-weight:900;color:#818cf8">89.9%</div>
    <div style="font-size:10px;color:#475569;margin-top:2px">CLIP · CIFAR-10</div>
  </div>
  <div style="flex:1;min-width:140px;background:#0f172a;border:1px solid #34d39933;
              border-radius:12px;padding:14px 16px">
    <div style="font-size:9px;color:#334155;text-transform:uppercase;letter-spacing:1px;
                margin-bottom:4px">Best Retrieval R@1</div>
    <div style="font-size:22px;font-weight:900;color:#34d399">86.8%</div>
    <div style="font-size:10px;color:#475569;margin-top:2px">BLIP · Flickr30k</div>
  </div>
  <div style="flex:1;min-width:140px;background:#0f172a;border:1px solid #fbbf2433;
              border-radius:12px;padding:14px 16px">
    <div style="font-size:9px;color:#334155;text-transform:uppercase;letter-spacing:1px;
                margin-bottom:4px">Largest Corpus</div>
    <div style="font-size:22px;font-weight:900;color:#fbbf24">1.8B</div>
    <div style="font-size:10px;color:#475569;margin-top:2px">ALIGN training pairs</div>
  </div>
  <div style="flex:1;min-width:140px;background:#0f172a;border:1px solid #f472b633;
              border-radius:12px;padding:14px 16px">
    <div style="font-size:9px;color:#334155;text-transform:uppercase;letter-spacing:1px;
                margin-bottom:4px">Prompt Ensembling</div>
    <div style="font-size:22px;font-weight:900;color:#f472b6">4×</div>
    <div style="font-size:10px;color:#475569;margin-top:2px">Templates averaged</div>
  </div>
</div>
"""

SIDEBAR_HTML = """
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
            padding:14px;margin-top:12px">
  <div style="font-size:10px;font-weight:700;color:#334155;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:10px">Supported Classes</div>
  <div style="display:flex;flex-wrap:wrap;gap:5px">
    <span style="background:#1e293b;color:#94a3b8;border-radius:6px;
                 padding:3px 8px;font-size:11px">✈️ Airplane</span>
    <span style="background:#1e293b;color:#94a3b8;border-radius:6px;
                 padding:3px 8px;font-size:11px">🚗 Automobile</span>
    <span style="background:#1e293b;color:#94a3b8;border-radius:6px;
                 padding:3px 8px;font-size:11px">🐦 Bird</span>
    <span style="background:#1e293b;color:#94a3b8;border-radius:6px;
                 padding:3px 8px;font-size:11px">🐱 Cat</span>
    <span style="background:#1e293b;color:#94a3b8;border-radius:6px;
                 padding:3px 8px;font-size:11px">🦌 Deer</span>
    <span style="background:#1e293b;color:#94a3b8;border-radius:6px;
                 padding:3px 8px;font-size:11px">🐶 Dog</span>
    <span style="background:#1e293b;color:#94a3b8;border-radius:6px;
                 padding:3px 8px;font-size:11px">🐸 Frog</span>
    <span style="background:#1e293b;color:#94a3b8;border-radius:6px;
                 padding:3px 8px;font-size:11px">🐴 Horse</span>
    <span style="background:#1e293b;color:#94a3b8;border-radius:6px;
                 padding:3px 8px;font-size:11px">🚢 Ship</span>
    <span style="background:#1e293b;color:#94a3b8;border-radius:6px;
                 padding:3px 8px;font-size:11px">🚚 Truck</span>
  </div>
</div>
<div style="background:#1a0f00;border:1px solid #fbbf2444;border-radius:10px;
            padding:10px 14px;margin-top:10px">
  <div style="font-size:11px;color:#fbbf24;font-weight:600">
    ⚡ First run loads weights (~30–60s per model). Subsequent runs are fast.
  </div>
</div>
"""

with gr.Blocks(title="CLIP OFF") as demo:
    gr.HTML(PAGE_HTML)

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            image_input = gr.Image(label="📸 Upload Image", type="numpy", height=260)
            model_select = gr.CheckboxGroup(
                choices=["CLIP ViT-B/16", "BLIP-base", "ALIGN-base", "FLAVA-full"],
                value=["CLIP ViT-B/16", "BLIP-base", "ALIGN-base", "FLAVA-full"],
                label="🤖 Select Models",
            )
            run_btn = gr.Button("⚡  Run Classification", variant="primary")
            gr.HTML(SIDEBAR_HTML)

        with gr.Column(scale=3):
            output_html = gr.HTML(
                label="🏆 Live Predictions",
                value=empty_state("Upload an image and click Run Classification")
            )

    run_btn.click(fn=predict, inputs=[image_input, model_select], outputs=output_html)
    image_input.change(fn=predict, inputs=[image_input, model_select], outputs=output_html)

demo.launch()
