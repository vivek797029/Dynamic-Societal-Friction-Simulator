# Google Colab Training Guide — 100 Unit Budget Edition

## Overview

This guide explains the completely rewritten and optimized `train_on_colab.ipynb` notebook designed for **strict 100 compute unit budgets** on Google Colab.

The new notebook includes:
- Real-time budget tracking with visual dashboards
- Automatic GPU detection and config optimization
- Smart checkpointing every 50 steps (instead of 100)
- Budget-aware training with auto-stop at 90% consumption
- Easy resumption from Colab disconnects
- Google Drive integration for checkpoint persistence

---

## Compute Unit Costs (2026 Pricing)

| GPU | Cost/Hour | 100 Units = | Epochs (9) | Status |
|-----|-----------|-------------|-----------|--------|
| **L4** | **2.35** | **42.6h** | ✅ 8-10 | **RECOMMENDED** |
| V100 | 2.70 | 37.0h | 8-9 | Good |
| T4 | 1.67 | 59.9h | 12+ | Slow, free tier limits |
| A100 40GB | 7.35 | 13.6h | 3-4 | Fast but tight |
| A100 80GB | 11.0 | 9.1h | 2-3 | Very expensive |
| H100 | 15.0 | 6.7h | 1-2 | Prohibitively expensive |

### Recommendation for 100-Unit Budget
**Use L4 GPU** for the best combination of:
- Cost efficiency (2.35 units/hour)
- Speed (good for Mistral-7B)
- Training duration (42+ hours allows 8-10 epochs)
- Safety margin (room for variations)

---

## Key Features of the New Notebook

### 1. BudgetTracker Class
Located in **Section 1**, automatically:
- Detects your GPU type (L4, A100, T4, V100, H100)
- Calculates hourly cost based on pricing table
- Tracks real-time consumption
- Warns when approaching limits
- Saves session logs to Google Drive

**Example output:**
```
🟢 BUDGET STATUS:
   Elapsed: 5.5h (330m)
   Consumed: 12.9 / 100 units (12.9%)
   Remaining: 87.1 units (37.1h)
```

### 2. GPU-Specific Auto-Configuration
Located in **Section 3**, automatically adjusts:

**For L4:**
```
- Epochs: 9
- Batch size: 2 × 16 = 32 effective
- Seq length: 2048
- LoRA rank: 64
- Save interval: 50 steps (frequent!)
```

**For A100:**
```
- Epochs: 4 (maximize training time)
- Batch size: 4 × 8 = 32 effective
- Seq length: 3072
- LoRA rank: 96
- Save interval: 50 steps
```

**For T4/V100:**
```
- Epochs: 12 (maximize budget usage)
- Batch size: 1 × 32 = 32 effective
- Seq length: 2048
- LoRA rank: 48 (memory optimized)
- FP16 mode (no bf16)
```

### 3. Smart Checkpointing
- **Saves every 50 steps** (was 100) for safety
- **Keeps only 5 recent checkpoints** to save space
- **Syncs to Google Drive immediately** after each save
- **Easy resume** even after complete Colab session restart
- **Budget-aware:** You can stop anytime and resume later

### 4. Budget Dashboard
Located in **Section 7**, call `plot_budget_dashboard()` to see:
- Pie chart of budget consumed vs. remaining
- Time estimation bar chart
- Detailed status with warnings
- Updated in real-time

### 5. Auto-Stop at 90%
Training automatically saves and exits when:
```python
budget_tracker.get_budget_percentage_used() >= 90
```
This prevents accidental overages while maximizing training time.

### 6. Training Resume
The notebook auto-detects and resumes from:
1. Local checkpoints (fastest)
2. Google Drive backup (if local deleted)
3. Starts fresh if no checkpoints found

To resume after disconnect:
1. Re-run Cell 1-4 (GPU setup)
2. Re-run Section 6 (training cell)
3. Notebook auto-resumes from latest checkpoint

---

## Section-by-Section Walkthrough

### Section 1: Initialize Budget Tracker & Check GPU
```python
# Auto-detects GPU type
# Mounts Google Drive
# Initializes BudgetTracker with pricing
```
**Time:** ~2 minutes

### Section 2: Clone Repo & Install Dependencies
```python
# Clones the DSFS repository
# Installs all dependencies
# Installs Flash Attention (for A100/H100)
```
**Time:** 3-5 minutes

### Section 3: GPU-Specific Config
```python
# Reads model_config.yaml
# Adjusts for detected GPU
# Optimizes batch size, epochs, LoRA rank
# Saves updated config
```
**Time:** ~30 seconds

### Section 4: Generate Training Data
```python
# Checks if data already exists
# Generates 50K+ training samples
# Augments with perspective flips, severity scaling
```
**Time:** 2-5 minutes (or instant if cached)

### Section 5: W&B Setup (Optional)
```python
# Disabled by default (faster)
# Uncomment to enable Weights & Biases logging
```
**Time:** ~10 seconds

### Section 6: TRAIN!
```python
# Main training loop
# Auto-resumes from checkpoint
# Syncs to Google Drive every 50 steps
# Budget tracking with warnings
# Auto-stops at 90% budget
```
**Time:** Depends on GPU and budget
- L4 with 100 units: ~42 hours
- A100 with 100 units: ~13.5 hours

### Section 7: Budget Dashboard
```python
# Visual monitoring of budget consumption
# Time estimates
# Detailed status with warnings
```
**Time:** ~1 second (call anytime)

### Section 8: Model Evaluation
```python
# Loads final trained adapter
# Tests with sample friction scenario
# Shows model output
```
**Time:** ~30 seconds

### Section 9: Export & Package
```python
# Copies adapter to Google Drive
# Creates model summary JSON
# Shows training statistics
# Downloads ready for use
```
**Time:** ~1 minute

---

## Step-by-Step Usage

### First Time Setup
1. **Open notebook** in Google Colab
2. **Change runtime to GPU:**
   - Click "Runtime" → "Change runtime type"
   - Select "T4 GPU" (or better if available)
   - Click "Save"
3. **Run Section 1** (GPU detection + budget tracker)
4. **Run Section 2** (clone + install)
5. **Run Section 3** (auto-config)
6. **Run Section 4** (generate data)
7. **Run Section 5** (W&B setup - optional)
8. **Run Section 6** (TRAIN!)

### Resume After Disconnect
1. **Re-run Section 1** (GPU setup)
2. **Run Section 6** directly - it will resume!
3. Check budget status with Section 7

### Check Budget Status Anytime
```python
# In any cell:
budget_tracker.print_status()

# Or with visuals:
plot_budget_dashboard()
```

---

## Budget Optimization Tips

### Tip 1: Check GPU Before Starting
The notebook auto-detects GPU in Section 1. If you don't see your preferred GPU (L4 recommended):
```
Runtime → Change runtime type → GPU
```
You may need to switch sessions to get L4 (L4 is less in-demand than T4).

### Tip 2: Monitor Budget Regularly
The notebook warns at:
- 🟡 50% budget consumed
- 🟠 75% budget consumed
- 🔴 90% budget consumed (auto-stop triggers)

Call `plot_budget_dashboard()` to visualize at any time.

### Tip 3: Save Checkpoints to Drive
The notebook auto-syncs every 50 steps. Your checkpoints are safe even if:
- Colab session disconnects
- GPU runs out of hours
- Training is manually stopped

You can download from: `/content/drive/MyDrive/dsfs-checkpoints/`

### Tip 4: Optimize Batch Size
The notebook auto-configures, but you can manually adjust in Section 3:
```yaml
# For L4, to train faster (might exceed budget):
per_device_train_batch_size: 4      # instead of 2
gradient_accumulation_steps: 8      # instead of 16

# For L4, to save budget (slower training):
per_device_train_batch_size: 1      # instead of 2
gradient_accumulation_steps: 32     # instead of 16
```

### Tip 5: Skip Evaluation if Running Tight on Budget
If you're approaching 90% budget, you can disable eval in Section 3:
```yaml
training:
  eval_steps: null  # or 0
  eval_strategy: "no"
```
This saves ~20-30% on training time.

---

## Troubleshooting

### Problem: GPU Not Detected
**Solution:**
```
Runtime → Change runtime type → GPU (select T4 minimum)
```

### Problem: "No space left on device"
**Solution:** The notebook runs out of disk space around 10GB+ of checkpoints.
```python
# In Section 3, reduce save_total_limit:
cfg['training']['save_total_limit'] = 2  # Keep only 2 checkpoints
```

### Problem: Out of Memory (OOM)
**Solution:** The auto-config handles this, but if you see OOM errors:
```python
# In Section 3, reduce batch size further:
cfg['training']['per_device_train_batch_size'] = 1
cfg['training']['gradient_accumulation_steps'] = 64
```

### Problem: Training is too slow
**Solution:** Switch to L4 GPU (faster than T4) or A100 (fastest):
```
Runtime → Change runtime type → GPU (select L4 or A100 if available)
```

### Problem: Budget tracker shows wrong GPU
**Solution:** The auto-detection looks for "L4", "A100", "T4", "V100", "H100" in GPU name.
```python
# Check detected type:
print(GPU_TYPE)

# If wrong, manually override (in any cell):
# budget_tracker.gpu_type = "L4"
```

---

## Output Files

After training completes, you'll have:

### Local (in Colab session)
- `outputs/checkpoints/final_adapter/` - Trained LoRA adapter
- `outputs/checkpoints/checkpoint-XXX/` - Intermediate checkpoints
- `outputs/model_summary.json` - Training metadata
- `budget_log.json` - Budget tracking log

### Google Drive
- `/MyDrive/dsfs-checkpoints/final_adapter/` - Final adapter backup
- `/MyDrive/dsfs-checkpoints/budget-log.json` - Budget log backup
- `/MyDrive/dsfs-checkpoints/checkpoint-XXX/` - Checkpoint backups

### Download Models
```
1. Go to Google Drive
2. Right-click on dsfs-checkpoints
3. Click "Download" (creates zip file)
4. Extract locally
5. Use adapter with: FrictionLLM(adapter_path="./final_adapter")
```

---

## Configuration Files Modified

The notebook auto-modifies `configs/model_config.yaml`:

**Original (full 15 epochs on A100):**
```yaml
training:
  num_train_epochs: 15
  per_device_train_batch_size: 4
  save_steps: 100
  max_seq_length: 4096
lora:
  r: 128
```

**After Auto-Config for L4:**
```yaml
training:
  num_train_epochs: 9
  per_device_train_batch_size: 2
  save_steps: 50
  max_seq_length: 2048
lora:
  r: 64
```

You can manually edit `configs/model_config.yaml` to fine-tune further.

---

## Advanced Usage

### Monitor Loss Curves
To plot training loss in real-time:
```python
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Load trainer_state.json from the training output
state_file = Path("outputs/checkpoints/trainer_state.json")
with open(state_file) as f:
    state = json.load(f)

# Extract loss history
if 'log_history' in state:
    steps = [h.get('step', 0) for h in state['log_history']]
    losses = [h.get('loss', 0) for h in state['log_history']]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, losses, linewidth=2)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.show()
```

### Manual Budget Check
```python
# At any point during training:
elapsed_h = budget_tracker.get_elapsed_hours()
consumed = budget_tracker.get_units_consumed()
remaining = budget_tracker.get_units_remaining()
pct = budget_tracker.get_budget_percentage_used()

print(f"Elapsed: {elapsed_h:.1f}h")
print(f"Consumed: {consumed:.1f} / 100 units ({pct:.1f}%)")
print(f"Remaining: {remaining:.1f} units")
```

### Force Stop Training
```python
# If training is using too much budget:
# 1. Click "Interrupt execution" button in the notebook
# 2. Budget log will auto-save
# 3. Re-run Section 6 to resume later
```

---

## Expected Outcomes

### For L4 GPU (Recommended)
- **Duration:** ~42 hours (within budget)
- **Epochs:** 9 full passes over training data
- **Quality:** Good. Multiple epochs improve performance
- **Cost:** Exactly ~100 units if training completes
- **Checkpoints:** ~8 saved (every ~100 steps)

### For A100 40GB
- **Duration:** ~13.5 hours (tight budget)
- **Epochs:** 4 full passes
- **Quality:** Moderate. Fast but fewer epochs
- **Cost:** Exactly ~100 units
- **Checkpoints:** ~6 saved

### For T4 (Free Tier)
- **Duration:** ~60 hours (very long)
- **Epochs:** 12 full passes
- **Quality:** Very good. Many epochs help
- **Cost:** Exactly ~100 units
- **Note:** T4 has usage limits in free tier, may disconnect

---

## Summary

The new notebook is a **complete rewrite optimized for 100-unit budgets:**

✅ **Easy Setup** - Auto-detects GPU and configures everything
✅ **Budget Safe** - Real-time tracking with 90% auto-stop
✅ **Crash Proof** - Resume from any disconnect via Google Drive
✅ **User Friendly** - Clear sections with emoji markers
✅ **Well Documented** - Each cell explains what it does
✅ **Flexible** - Easy to tweak for different GPU types
✅ **Reliable** - Comprehensive error handling

**Next steps:** Open the notebook in Colab and follow Section 1!
