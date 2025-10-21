# Data Requirements for Research Questions

## Overview
This document specifies all metrics, visualizations, and data that must be collected during training and inference to answer our research questions.

---

## 1. Latent Space Controllability

### Training-Time Data
```python
{
    "latent_statistics": {
        "mean": [batch_size, latent_dim],           # Latent means
        "logvar": [batch_size, latent_dim],         # Latent log-variances
        "samples": [batch_size, latent_dim],        # Sampled latents
        "sdf_values": [batch_size, num_observers]   # Per-observer SDF values
    },
    "manifold_metrics": {
        "sdf_mean": float,                          # Average SDF across observers
        "sdf_std": float,                           # SDF standard deviation
        "manifold_distance": [batch_size]           # Distance to manifold
    }
}
```

### Inference-Time Data
```python
{
    "latent_traversal": {
        "start_latent": [latent_dim],
        "end_latent": [latent_dim],
        "interpolated_latents": [num_steps, latent_dim],
        "interpolated_sdfs": [num_steps, num_observers],
        "generated_images": [num_steps, 3, H, W],
        "reconstruction_quality": [num_steps]       # Per-step quality metrics
    },
    "sdf_guided_generation": {
        "target_sdf": float,
        "achieved_sdf": float,
        "generated_image": [3, H, W],
        "latent_code": [latent_dim]
    }
}
```

**Required Functions**:
- `interpolate_latents(z1, z2, num_steps)` → latent path
- `evaluate_sdf_along_path(latent_path)` → SDF values
- `generate_at_sdf_level(target_sdf)` → image + achieved SDF
- `visualize_latent_manifold()` → 3D plot with SDF coloring

---

## 2. Observer Interpretation

### Training-Time Data
```python
{
    "observer_metrics": {
        "attention_weights": [batch_size, num_observers],  # Softmax attention
        "attention_entropy": float,                         # H(p) = -Σ p log p
        "observer_features": [batch_size, num_observers, projection_dim],
        "observer_sdf": [batch_size, num_observers, 1],
        "feature_diversity": float                          # Pairwise cosine sim
    },
    "per_observer_stats": {
        "observer_0": {
            "mean_attention": float,
            "std_attention": float,
            "activation_magnitude": float,
            "feature_norm": float
        },
        # ... repeat for all observers
    }
}
```

### Inference-Time Data
```python
{
    "observer_activation_maps": {
        "input_image": [3, H, W],
        "observer_0_activations": [projection_dim],
        # ... for each observer
        "grad_cam_maps": [num_observers, H, W]            # Gradient-based saliency
    },
    "observer_ablation": {
        "full_model_recon": [3, H, W],
        "without_observer_0": [3, H, W],
        # ... for each observer
        "reconstruction_quality_drop": [num_observers]     # Impact of removing each
    },
    "feature_importance": {
        "observer_0_top_features": [top_k, projection_dim],
        # ... for each observer
        "feature_overlap_matrix": [num_observers, num_observers]  # Similarity matrix
    }
}
```

**Required Functions**:
- `get_observer_activations(image)` → per-observer features
- `compute_grad_cam(image, observer_id)` → saliency map
- `ablate_observer(image, observer_id)` → reconstruction without observer
- `compute_feature_importance(observer_id)` → ranked features
- `visualize_observer_specialization()` → radar plot of observer roles

---

## 3. Baseline Comparison

### Training-Time Data (All Models)
```python
{
    "training_curves": {
        "epoch": int,
        "train_loss": float,
        "val_loss": float,
        "test_loss": float,
        "components": {
            "reconstruction": float,
            "kl_divergence": float,
            # Model-specific losses below:
            "sdf_consistency": float,      # Ours only
            "eikonal": float,              # Ours only
            "diversity": float             # Ours only
        }
    },
    "generalization_metrics": {
        "train_test_gap": float,
        "overfitting_score": float,
        "epochs_to_convergence": int
    }
}
```

### Evaluation Data (All Models)
```python
{
    "reconstruction_metrics": {
        "mse": float,
        "psnr": float,
        "ssim": float,
        "lpips": float                     # Perceptual similarity
    },
    "generation_quality": {
        "fid": float,                      # Fréchet Inception Distance
        "is": float,                       # Inception Score
        "sample_diversity": float
    },
    "latent_space_metrics": {
        "mig": float,                      # Mutual Information Gap
        "sap": float,                      # Separated Attribute Predictability
        "dci": float,                      # Disentanglement-Completeness-Informativeness
        "interpretability_score": float    # Custom: SDF correlation
    }
}
```

**Required Functions**:
- `train_vanilla_vae()` → baseline model
- `train_beta_vae(beta)` → β-VAE baseline
- `compute_disentanglement_metrics(model)` → MIG, SAP, DCI
- `compute_fid(real_images, generated_images)` → FID score
- `compare_latent_spaces([models])` → visualization + metrics

---

## 4. Scalability Analysis

### Profiling Data
```python
{
    "training_profile": {
        "batch_size": int,
        "time_per_epoch": float,
        "time_per_batch": float,
        "gpu_memory_allocated": float,
        "gpu_memory_reserved": float,
        "cpu_memory": float
    },
    "loss_component_timing": {
        "reconstruction_time": float,
        "kl_time": float,
        "sdf_consistency_time": float,
        "eikonal_time": float,              # O(n²) component
        "diversity_time": float,
        "total_forward_time": float,
        "total_backward_time": float
    },
    "complexity_analysis": {
        "num_parameters": int,
        "flops_forward": int,
        "flops_backward": int,
        "memory_footprint_mb": float
    }
}
```

### Ablation Data
```python
{
    "eikonal_ablation": {
        "with_eikonal": {
            "final_loss": float,
            "convergence_epoch": int,
            "training_time_hours": float,
            "gpu_memory_peak_mb": float
        },
        "without_eikonal": {
            "final_loss": float,
            "convergence_epoch": int,
            "training_time_hours": float,
            "gpu_memory_peak_mb": float
        },
        "performance_impact": {
            "reconstruction_quality_delta": float,
            "training_speedup": float,
            "memory_saved_mb": float
        }
    }
}
```

**Required Functions**:
- `profile_training(model, batch_sizes)` → timing + memory data
- `ablation_study(loss_components)` → performance with/without each
- `analyze_complexity(model)` → FLOPs, params, memory

---

## 5. Generalization Mechanism

### Ablation Studies Data
```python
{
    "curriculum_ablation": {
        "with_curriculum": {
            "train_losses": [num_epochs],
            "val_losses": [num_epochs],
            "final_train_test_gap": float,
            "convergence_stability": float
        },
        "without_curriculum": {
            "train_losses": [num_epochs],
            "val_losses": [num_epochs],
            "final_train_test_gap": float,
            "convergence_stability": float
        }
    },
    "loss_component_ablation": {
        "full_model": {...},
        "no_sdf_consistency": {...},
        "no_eikonal": {...},
        "no_diversity": {...},
        "no_curriculum": {...}
    },
    "diversity_impact": {
        "diversity_weight_0.0": {...},
        "diversity_weight_0.1": {...},   # Original
        "diversity_weight_0.2": {...},
        "diversity_weight_0.5": {...}
    }
}
```

### Overfitting Analysis
```python
{
    "overfitting_metrics": {
        "epoch": [num_epochs],
        "train_loss": [num_epochs],
        "val_loss": [num_epochs],
        "test_loss": [num_epochs],
        "train_val_gap": [num_epochs],
        "train_test_gap": [num_epochs],
        "gap_variance": float,              # Lower = more stable
        "max_gap": float,                   # Peak overfitting
        "final_gap": float                  # End-of-training overfitting
    },
    "regularization_effectiveness": {
        "observer_diversity_correlation": float,  # Diversity vs generalization
        "sdf_consistency_correlation": float,     # Consistency vs generalization
        "attention_entropy_over_time": [num_epochs]
    }
}
```

**Required Functions**:
- `run_ablation_study(config_variations)` → comparative results
- `analyze_generalization(train_history, val_history)` → overfitting metrics
- `correlation_analysis(diversity_loss, generalization_gap)` → relationships

---

## Summary: Required Code Additions

### In `src/training/trainer.py`:
```python
class MultiPerspectiveTrainer:
    def __init__(self, ...):
        self.metrics_logger = MetricsLogger()  # NEW

    def train_epoch(self, ...):
        # Existing training code

        # ADD: Log detailed metrics
        self.metrics_logger.log({
            'latent_statistics': {...},
            'observer_metrics': {...},
            'training_profile': {...}
        })

    def validate(self, ...):
        # ADD: Collect validation metrics
        return comprehensive_metrics
```

### New Files Needed:
1. `src/analysis/latent_controllability.py` - Q1 analysis
2. `src/analysis/observer_interpretation.py` - Q2 analysis
3. `src/analysis/baseline_comparison.py` - Q3 analysis
4. `src/analysis/scalability_profiling.py` - Q4 analysis
5. `src/analysis/generalization_analysis.py` - Q5 analysis
6. `src/utils/metrics_logger.py` - Centralized metric collection
7. `src/utils/visualization.py` - Publication-quality plots

### Jupyter Notebooks Needed:
1. `notebooks/01_latent_space_exploration.ipynb`
2. `notebooks/02_observer_specialization.ipynb`
3. `notebooks/03_baseline_comparisons.ipynb`
4. `notebooks/04_scalability_analysis.ipynb`
5. `notebooks/05_generalization_study.ipynb`
6. `notebooks/06_paper_figures.ipynb`

---

## Next Steps

1. **Extract and refactor** existing Medical_one.py into modular components
2. **Add metric collection** hooks throughout training code
3. **Implement analysis functions** for each research question
4. **Create baseline models** (Vanilla VAE, β-VAE)
5. **Run systematic experiments** to collect all required data
6. **Generate publication figures** from collected data
7. **Write paper** with empirical support for all claims
