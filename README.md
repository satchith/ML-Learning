# ML Learning

My working notebooks and exercises while learning machine learning and neural networks, with the longer-term aim of building machine-learning
interatomic potentials (MLIPs) for atomistic simulation.

This is a **learning log, not a portfolio** - the code here is for practice and understanding. My polished, project-level work lives in separate repos
(see my [pinned repositories](https://github.com/satchith)).

## Focus

I'm following a structured plan that moves from ML fundamentals toward the physics of interatomic potentials: gradient descent and backpropagation,
automatic differentiation, PyTorch, and the descriptor/symmetry-function ideas behind neural-network potentials. The thread I care about most is the
energy to forces connection (F = -∂E/∂r by autodiff), which is the conceptual core of Deep Potential.

## Notes

Each folder has a short note on what I set out to do, what I learned, and what's still rough. I'm prioritizing understanding over polish here.

## References

- Géron, *Hands-On Machine Learning* (2025)
- Nielsen, *Neural Networks and Deep Learning*
- Behler & Parrinello, *PRL* 98, 146401 (2007)
