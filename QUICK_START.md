# Quick Start — 100 Unit Budget Colab Training

## TL;DR - In 30 Seconds

1. Open `notebooks/train_on_colab.ipynb` in Google Colab
2. Change Runtime to **L4 GPU** (recommended)
3. **Run all cells in order** (Sections 1-6)
4. Training auto-stops at 90% budget (~42 hours on L4)
5. Download model from Google Drive

---

## Budget Overview

**For 100 Compute Units:**

```
L4 GPU:       42 hours  ✅ RECOMMENDED (best value)
A100:         13.5 hours (fast, but few epochs)
T4:           60 hours  (slow, free tier limited)
```

---

## Notebook Sections

| Section | What It Does | Time |
|---------|------------|------|
| **1** | 🔧 Setup GPU + Budget Tracker | 2 min |
| **2** | 📦 Clone repo + Install deps | 3-5 min |
| **3** | ⚙️ Auto-configure for your GPU | 30 sec |
| **4** | 📊 Generate training data | 2-5 min |
| **5** | 🎯 Setup W&B (optional) | 10 sec |
| **6** | 🚀 **START TRAINING** | 13-42 hours |
| **7** | 📈 Budget dashboard | 1 sec |
| **8** | ✅ Test final model | 30 sec |
| **9** | 📦 Export + Download | 1 min |

**Total prep time before training: ~10 minutes**

---

## Pre-Training Checklist

- [ ] Open notebook in Google Colab
- [ ] Click **Runtime → Change runtime type → GPU**
- [ ] Select **L4** (or A100 if available)
- [ ] Click **Save**
- [ ] Start with Section 1

---

## During Training

**Check budget anytime:**
```python
# Call in any cell:
budget_tracker.print_status()

# Or visualize:
plot_budget_dashboard()
```

**Status indicators:**
- 🟢 Green: 0-50% budget used (good)
- 🟡 Yellow: 50-75% budget used (watch it)
- 🟠 Orange: 75-90% budget used (warning)
- 🔴 Red: 90%+ budget used (auto-stops)

---

## If Colab Disconnects

1. **Re-run Section 1** (GPU setup)
2. **Re-run Section 6** (training)
3. ✅ Notebook auto-resumes from last checkpoint
4. Done!

---

## After Training

**Final model location:**
- Colab: `outputs/checkpoints/final_adapter/`
- Google Drive: `/MyDrive/dsfs-checkpoints/final_adapter/`

**Download from Drive:**
1. Open Google Drive
2. Navigate to `/MyDrive/dsfs-checkpoints/`
3. Right-click → Download
4. Extract zip file

---

## Cost Calculator

**Input your training time:**
```
Hours × GPU_rate = Units consumed

Example (L4):
40 hours × 2.35 units/hour = 94 units ✅ (within budget)
```

**GPU rates (units/hour):**
- L4: **2.35** (recommended)
- V100: 2.70
- T4: 1.67 (slow)
- A100: 7.35 (expensive)

---

## Common Issues & Fixes

**GPU Not Available?**
→ Try switching sessions or selecting T4

**Out of Memory?**
→ Auto-config handles this. If error, reduce batch size in Section 3

**Training Too Slow?**
→ Switch to L4 or A100 GPU (faster than T4)

**Ran out of budget?**
→ Download checkpoint from Google Drive and finish locally

---

## Model Specs

```
Base Model:    Mistral-7B-Instruct-v0.3
Fine-tune:     QLoRA (4-bit quantized)
LoRA Rank:     48-96 (auto-configured)
Data:          50K+ friction scenario samples
Epochs:        3-12 (depending on GPU)
```

---

## What Gets Auto-Configured

The notebook detects your GPU and automatically adjusts:

✅ Number of training epochs
✅ Batch size & gradient accumulation
✅ Sequence length
✅ LoRA rank (how powerful the adapter is)
✅ Save checkpoint frequency
✅ Precision (bf16 for A100, fp16 for T4)

**No manual config needed!**

---

## Files Created

After training:
```
outputs/checkpoints/
  ├── final_adapter/          ← Your trained model
  ├── checkpoint-500/         ← Backup checkpoints
  ├── checkpoint-1000/
  └── trainer_state.json      ← Training metadata

outputs/model_summary.json    ← Training stats

budget_log.json              ← Budget tracking
```

---

## Key Features

🎯 **Budget-Safe**
- Real-time tracking
- Auto-stops at 90%
- No surprise overages

🔄 **Crash-Proof**
- Resume from Google Drive
- Works after Colab disconnect
- Saves every 50 steps

📊 **Transparent**
- Visual dashboard
- Detailed logs
- Training metrics

⚡ **Fast Setup**
- Auto-GPU detection
- Auto-config for your GPU
- One-click training

---

## Support

**Questions?** Check `COLAB_BUDGET_GUIDE.md` for:
- Detailed troubleshooting
- Advanced configurations
- Monitoring & tips
- Resume instructions

---

## Next Step

**Open the notebook and run Section 1!**

```
URL: notebooks/train_on_colab.ipynb
```

Good luck! 🚀
