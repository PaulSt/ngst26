---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Embedded Trefftz

Classical Trefftz methods need an explicit local basis of functions satisfying the PDE. 
These are difficult to construct. 
Embedded Trefftz works differently:

1. start with a standard discontinuous polynomial space,
2. assemble **elementwise** a local version of the PDE,
3. compute the **elementwise** nullspace of that residual,
4. assemble the usual DG formulation on that nullspace.

Moreover, we relax the Trefftz condition in a way that generalizes to operators that do not have a kernel in the polynomials.

:::{admonition} The space we are looking for!
$$
\mathbb T_h^p(K)
=
\{v_h\in \mathbb P^p(K):
(\mathcal L v_h,q_h)_K=0
\quad \forall q_h\in Q_h(K)\}.
$$
:::

## Constructing an Embedding

The embedding represents the unknown Trefftz basis inside the standard discontinuous polynomial basis $\{\phi_i\}_{i=1}^N$.

$$
\psi_j=\sum_{i=1}^N \mathbf{T}_{ij}\phi_i,
\qquad
\mathbf T\in\mathbb R^{N\times M},\qquad M\ll N.
$$

If $A$ is the ordinary DG matrix on the full polynomial space, then we solve is the projected system

$$
\mathbf T^T \mathbf A \mathbf T\,u_T = \mathbf T^T b.
$$

The embedding is constructed from the residual equations. 

$$
(\mathbf W_K)_{ji}=(\mathcal L_K\phi_i,\xi_j)_K .
$$

The local Trefftz coefficients are the nullspace of $W_K$. 
Numerically, `TrefftzEmbedding` computes that nullspace element by element with an SVD (or equivalently a QR):

$$
\boxed{
{\color{red}\mathbf{T}}
=
\ker(\mathbf{W}),
\qquad
\mathbf W_{ji}
=
\sum_{K\in\mathcal{T}_h}
\left\langle
\mathcal{L}\,{\color{blue}\phi_i},
{\color{purple}\xi_j}
\right\rangle_K
}
$$

On each mesh element, use an SVD
$\textcolor{gray}{\text{(or QR decomposition)}}$:

$$
\left.\mathbf{W}\right|_K
=
{\color{gray}
\underbrace{
\begin{pmatrix}
| &        & | \\
\mathbf{u}_1 & \cdots & \mathbf{u}_L \\
| &        & |
\end{pmatrix}
}_{\mathbf{U}_K}
}
\;
\underbrace{
\begin{pmatrix}
\sigma_1 &        &          & {\color{green}0} \\
         & \ddots &          & \vdots \\
         &        & \sigma_L & {\color{green}0}
\end{pmatrix}
}_{\boldsymbol{\Sigma}_K}
\;
\underbrace{
\begin{pmatrix}
- & \mathbf{v}_1^{T} & - \\
  & \vdots            &   \\
- & \mathbf{v}_L^{T} & - \\[0.4em]
{\color{red}-} & {\color{red}\mathbf{v}_{L+1}^{T}} & {\color{red}-} \\
  & {\color{red}\vdots} &   \\
{\color{red}-} & {\color{red}\mathbf{v}_N^{T}} & {\color{red}-}
\end{pmatrix}
}_{\mathbf{V}_K^{T}} .
$$

Hence, the local embedding is given by

$$
{\color{red}
\mathbf{T}_K
=
\begin{pmatrix}
| &        & | \\
\mathbf{v}_{L+1} & \cdots & \mathbf{v}_N \\
| &        & | 
\end{pmatrix}.
}
$$

## Conditioning of the embedded Trefftz method.

$$
\boxed{
\kappa_2\!\left(
{\color{red}\mathbf{T}}^{T}
{\color{blue}\mathbf{A}}
{\color{red}\mathbf{T}}
\right)
\leq
\kappa_2\!\left(
{\color{blue}\mathbf{A}}
\right)
}
$$

```{code-cell} ipython3
:tags: [hide-input]

from ngsolve import *
from netgen.occ import *
from ngsolve_book import Draw
from ngstrefftz import *
SetNumThreads(4)

import matplotlib.pyplot as plt
import scipy.sparse as sp

Lap = lambda u: sum(Trace(u.Operator("hesse")))
```

## A first Embedding

Since

$$
-\Delta:\mathbb P^p(K)\to \mathbb P^{p-2}(K),
$$

choosing $Q(K)=\mathbb P^{p-2}(K)$ yields

$$
v\in \mathbb T^p(K)
\iff
(-\Delta v,q)_K=0 \quad \forall q\in \mathbb P^{p-2}(K)
\iff
\Delta v=0.
$$

```{code-cell} ipython3
mesh = Mesh(unit_square.GenerateMesh(maxh=0.35))
p = 4

base = L2(mesh, order=p, dgjumps=True)
test = L2(mesh, order=p - 2, dgjumps=True)

u = base.TrialFunction()
q = test.TestFunction()

op = (-Lap(u)) * q * dx
emb = TrefftzEmbedding(op)

print(f"full polynomial dofs: {base.ndof}")
print(f"embedded Trefftz dofs: {emb.GetEmbedding().shape[1]}")
```

The embedded unknown lives in the small Trefftz space. `emb.Embed(...)`
converts it back to the polynomial space for visualization or post-processing.

```{code-cell} ipython3
ut = emb.GetEmbedding().CreateRowVector()
up = GridFunction(base, multidim=len(ut))

for i, vec in enumerate(up.vecs):
    ut[:] = 0
    ut[i] = 1
    vec.data = emb.Embed(ut)

Draw(up,mesh,"basis",animate=True,interpolate_multidim=False,min=-1, max=1, deformation=True, scale=.3, euler_angles=[-70,0.4,2],)
```

## Setting up the global embedded DG system

We can use again the standard SIPDG

$$
\begin{aligned}
a_h(u,v)
&=
\int_{\Omega} \nabla u \cdot \nabla v \,\mathrm{d}x
-
\int_{\mathcal{F}_h^{\mathrm{int}}}
\left(
\{\!\{\nabla u\}\!\} \cdot [\![ v ]\!]
+
\{\!\{\nabla v\}\!\} \cdot [\![ u ]\!]
-
\frac{\alpha p^2}{h}
[\![ u ]\!] \cdot [\![ v ]\!]
\right)
\,\mathrm{d}s
\\
&\quad
-
\int_{\mathcal{F}_h^{\mathrm{bnd}}}
\left(
n\cdot \nabla u\,v
+
n\cdot \nabla v\,u
-
\frac{\alpha p^2}{h}\,uv
\right)
\,\mathrm{d}s,
\\
\ell(v)
&=
\int_{\mathcal{F}_h^{\mathrm{bnd}}}
\left(
\frac{\alpha p^2}{h}\,gv
-
n\cdot\nabla v\,g
\right)
\,\mathrm{d}s .
\end{aligned}
$$

```{code-cell} ipython3
:tags: [hide-input]

def sip_laplace(fes, bndc, rhs=0, alpha=4):
    mesh = fes.mesh
    order = fes.globalorder
    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    u = fes.TrialFunction()
    v = fes.TestFunction()

    jump_u = (u - u.Other()) * n
    jump_v = (v - v.Other()) * n
    mean_dudn = 0.5 * (grad(u) + grad(u.Other()))
    mean_dvdn = 0.5 * (grad(v) + grad(v.Other()))

    a = BilinearForm(fes, symmetric=True)
    a += grad(u) * grad(v) * dx
    a += alpha * order**2 / h * jump_u * jump_v * dx(skeleton=True)
    a += (-mean_dudn * jump_v - mean_dvdn * jump_u) * dx(skeleton=True)
    a += alpha * order**2 / h * u * v * ds(skeleton=True)
    a += (-n * grad(u) * v - n * grad(v) * u) * ds(skeleton=True)

    f = LinearForm(fes)
    f += rhs * v * dx
    f += alpha * order**2 / h * bndc * v * ds(skeleton=True)
    f += -n * grad(v) * bndc * ds(skeleton=True)

    with TaskManager():
        a.Assemble()
        f.Assemble()
    return a, f
```

First, the explicit matrix projection mirrors the derivation:
```{code-cell} ipython3
exact = exp(x) * sin(y)

a_full, f_full = sip_laplace(base, exact)
T = emb.GetEmbedding()
TT = T.CreateTranspose()

with TaskManager():
    A_t = TT @ a_full.mat @ T
    b_t = TT * f_full.vec
    sol_t = A_t.Inverse(inverse="sparsecholesky") * b_t

gfu = GridFunction(base)
gfu.vec.data = T * sol_t

print(f"error: {sqrt(Integrate((gfu - exact) ** 2, mesh)):.3e}")

Draw(gfu, mesh, "projected embedded solve");
```


## EmbeddedTrefftzFES
For production code, `EmbeddedTrefftzFES` avoids assembling the larger full
matrix first:

```{code-cell} ipython3
tfes = EmbeddedTrefftzFES(emb)

a_emb, f_emb = sip_laplace(tfes, exact)
gfu_t = GridFunction(tfes)

with TaskManager():
    gfu_t.vec.data = a_emb.mat.Inverse(inverse="sparsecholesky") * f_emb.vec

gfu = GridFunction(base)
gfu.vec.data = emb.Embed(gfu_t.vec)

print(f"error: {sqrt(Integrate((gfu - exact) ** 2, mesh)):.3e}")
```


## Inhomogeneous Problems

For $\mathcal L u=f$ the unknown is split into a local particular part and a homogeneous Trefftz correction:

$$
u_h = u_{h,f} + \mathbf T u_T.
$$

The local equations determine $u_{h,f}$ locally, while the global DG solve determines only $u_T$:

$$
\mathbf T^T \mathbf A \mathbf T u_T = \mathbf T^T(b - \mathbf A u_{h,f}).
$$

```{code-cell} ipython3
p = 4

base = L2(mesh, order=p, dgjumps=True)
test = L2(mesh, order=p - 2, dgjumps=True)

u = base.TrialFunction()
q = test.TestFunction()

exact = sin(pi * x) * sin(pi * y)
rhs = 2 * pi**2 * exact

op = Lap(u) * q * dx
lop = -rhs * q * dx

emb = TrefftzEmbedding(op, lop)
up = emb.GetParticularSolution()

a, f = sip_laplace(base, exact, rhs=rhs)
T = emb.GetEmbedding()
TT = T.CreateTranspose()

with TaskManager():
    A_t = TT @ a.mat @ T
    b_t = TT * (f.vec - a.mat * up)
    sol_t = A_t.Inverse(inverse="sparsecholesky") * b_t

gfu = GridFunction(base)
gfu.vec.data = T * sol_t + up

print(f"Poisson embedded error: {sqrt(Integrate((gfu - exact) ** 2, mesh)):.3e}")
print(f"full dofs: {base.ndof}, Trefftz dofs: {T.width}")

Draw(gfu, mesh, "gfu");
```

## Local-Global Coupling

The embedded view also separates the global DG coupling from the local residual constraints. Recall the weak Trefftz space

$$
\mathbb T_h
=
\left\{
v_h\in V_h:
(\mathcal L_K v_h,q_h)_K=0
\quad \forall q_h\in Q_h(K),\ K\in\mathcal T_h
\right\}.
$$

For inhomogeneous problems, the algebraic problem can be written as a coupled local-global system: find $u_h\in V_h$ such that

$$
\begin{cases}
a_h(u_h,v_h)=\ell_h(v_h)
&\forall v_h\in \mathbb T_h,\\[0.4em]
(\mathcal L_K u_h,q_h)_K=(f,q_h)_K
&\forall q_h\in Q_h(K),\ K\in\mathcal T_h.
\end{cases}
$$

The crucial decomposition is

$$
V_h=\mathbb T_h\oplus \mathbb L_h.
$$

Here $\mathbb L_h$ is a local lifting complement. In the block picture below, the lower-left block is zero because Trefftz trial functions satisfy the local residual equations. 
The lower-right block is local and determines the lifting or particular part; the upper blocks are the globally coupled DG equations.

:::{figure} _static/tikz/global-system.svg
:width: 72%
:::
<!--
The analysis asks for two independent stability mechanisms: $a_h$ must be
well-posed on $\mathbb T_h$, and the local maps
$\mathcal L_K:\mathbb L_h(K)\to Q_h(K)'$ must control the lifting component,

$$
\|\mathcal L_K u_h\|_{Q_h(K)'}^2
\gtrsim
\|u_h\|_{V_h(K)}^2
\qquad \forall u_h\in\mathbb L_h(K).
$$

Together with residual continuity,

$$
\sum_{K\in\mathcal T_h}
\|\mathcal L_K u\|_{Q_h(K)'}^2
\lesssim
\|u\|_{\widetilde V_h}^2,
$$

this gives the expected quasi-best-approximation statement

$$
\|u-u_h\|_{V_h}
\lesssim
\inf_{v_h\in V_h}\|u-v_h\|_{\widetilde V_h}.
$$
-->


:::{admonition} API to remember
:class: important

`TrefftzEmbedding(op, lop)` creates the Trefftz embedding

`EmbeddedTrefftzFES(emb)` creates a embedded Trefftz space 

`emb.GetParticularSolution()` returns the element-wise particular solution

`emb.Embed(vec)` embeds a Trefftz function into the standard polynomial space

Everything else is ordinary NGSolve DG assembly.
:::
