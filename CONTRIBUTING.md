# Contributing to Multi-Perspective SDF-VAE Research

Thank you for your interest in contributing to this research project! This document provides guidelines for contributing.

## 🎯 Project Overview

This is an independent research project exploring interpretable latent representations through Signed Distance Functions in Variational Autoencoders. The project is open for community contributions, extensions, and improvements.

## 🤝 How to Contribute

### Reporting Issues

- **Bug Reports**: Open an issue with a clear description, steps to reproduce, and your environment details
- **Feature Requests**: Describe the feature, its motivation, and potential implementation approach
- **Research Questions**: Share ideas for new experiments or analyses

### Code Contributions

1. **Fork the Repository**
   ```bash
   git clone https://github.com/Ibzie/Multi-Perspective-SDF-VAE-Research.git
   cd Multi-Perspective-SDF-VAE-Research
   ```

2. **Set Up Development Environment**
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation as needed

5. **Test Your Changes**
   ```bash
   # Quick sanity check (2 epochs)
   python experiments/train_main_model.py --dataset celeba --epochs 2 --batch_size 32
   ```

6. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Provide a clear description of changes
   - Reference any related issues
   - Include results/visualizations if applicable

## 📋 Contribution Areas

### 🔬 Research Extensions

- **New Architectures**: Implement variations of the SDF-VAE architecture
- **Datasets**: Add support for new datasets (3D shapes, medical imaging, etc.)
- **Metrics**: Implement additional evaluation metrics
- **Ablation Studies**: Design and run new ablation experiments

### 💻 Code Improvements

- **Performance**: Optimize training speed or memory usage
- **Documentation**: Improve code comments and documentation
- **Testing**: Add unit tests or integration tests
- **Visualization**: Create new visualization tools

### 📊 Experiments

- **Reproducibility**: Verify and document reproduction of results
- **Comparisons**: Compare with other recent methods
- **Analysis**: Provide deeper analysis of model behavior

### 📝 Documentation

- **Tutorials**: Create tutorials for specific use cases
- **Examples**: Add example notebooks
- **Explanations**: Improve explanations of concepts

## 🎨 Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and modular

Example:
```python
def compute_sdf_loss(predictions: torch.Tensor, 
                     targets: torch.Tensor,
                     weight: float = 1.0) -> torch.Tensor:
    """
    Compute the SDF consistency loss.
    
    Args:
        predictions: Predicted SDF values [batch_size, ...]
        targets: Target SDF values [batch_size, ...]
        weight: Loss weight coefficient
        
    Returns:
        Weighted SDF loss
    """
    loss = F.mse_loss(predictions, targets)
    return weight * loss
```

## 🧪 Testing Guidelines

- Test on small datasets first (use `--epochs 2` for quick tests)
- Verify that training converges
- Check that outputs are reasonable
- Document any unexpected behavior

## 📄 Documentation Guidelines

- Update README.md if adding new features
- Add docstrings to new functions
- Include usage examples
- Document any new dependencies

## 🔍 Research Ethics

- Cite relevant prior work
- Be transparent about limitations
- Share negative results (they're valuable!)
- Respect data licenses and usage terms

## 💬 Communication

- **Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

All contributors will be acknowledged in the project. Significant contributions may warrant co-authorship on any resulting publications (to be discussed on a case-by-case basis).

## ❓ Questions?

Feel free to open an issue with the `question` label if you need clarification on anything!

---

Thank you for contributing to advancing research in interpretable deep learning! 🚀
