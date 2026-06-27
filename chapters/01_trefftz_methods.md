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

# What Are Trefftz Methods?

Trefftz methods build local approximation spaces from functions that already
satisfy the PDE on each element. 
For the Laplace equation this means harmonic polynomials,

$$
\mathbb T^p(K)
= \{v \in \mathbb P^p(K) : \Delta v = 0\}.
$$

In two dimensions a convenient harmonic basis begins as

$$
1,\quad x,\quad y,\quad xy,\quad x^2-y^2,\quad
x^3-3xy^2,\ldots
$$

so the local dimension drops from

$$
\dim \mathbb P^p(K)=\frac{(p+1)(p+2)}2
\qquad\hbox{to}\qquad
\dim \mathbb T^p(K)=2p+1.
$$

```{code-cell} ipython3
:tags: [hide-input]

from ngsolve import *
from netgen.occ import unit_square
from ngsolve_book import Draw
from ngstrefftz import *

import matplotlib.pyplot as plt
import scipy.sparse as sp

Lap = lambda u: sum(Trace(u.Operator("hesse")))
mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
```

## Laplace: Classical Trefftz-DG

The NGSTrefftz constructor receives the mesh, polynomial order, and the
equation key:

```{code-cell} ipython3
order = 6
fes = trefftzfespace(
    mesh,
    order=order,
    eq="laplace",
    dgjumps=True,
)

print(f"Trefftz dofs: {fes.ndof}")
print(f"Full polynomial dofs: {L2(mesh,order=order).ndof}")
```

The formulation used below is the symmetric interior penalty DG method. 

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

def dglap(fes, bndc, rhs=0, alpha=4):
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

We solve a harmonic boundary value problem with exact solution
$u(x,y)=e^x\sin(y)$.

```{code-cell} ipython3
exact = exp(x) * sin(y)

a, f = dglap(fes, exact)
gfu = GridFunction(fes)

with TaskManager():
    gfu.vec.data = a.mat.Inverse(inverse="sparsecholesky") * f.vec

err = sqrt(Integrate((gfu - exact) ** 2, mesh))
print(f"Trefftz L2 error: {err:.3e}")

Draw(gfu, mesh, "Laplace Trefftz solution");
```

While the coupling pattern stays the same, dofs are reduced:

```{code-cell} ipython3
import scipy.sparse as sp
import numpy as np
import matplotlib.pylab as plt
fes_poly = L2(mesh, order=order, dgjumps=True)
a2, _ = dglap(fes_poly, exact)

A1 = sp.csr_matrix(a.mat.CSR())
A2 = sp.csr_matrix(a2.mat.CSR())
fig = plt.figure(); ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122)
ax1.set_xlabel("Trefftz=True"); ax1.spy(A1,markersize=1); ax1.set(ylim=(a2.mat.height,0),xlim=(0,a2.mat.width))
ax2.set_xlabel("Trefftz=False"); ax2.spy(A2,markersize=1)
```

## Helmholtz: Plane-Wave Trefftz Spaces

For Helmholtz,

$$
-\Delta u - \omega^2 u = 0,
$$

the kernel is non-polynomial. 
NGSTrefftz provides plane-wave spaces:

$$
\mathbb T^p_{\omega}(K)
= \operatorname{span}\left\{
e^{-i\omega d_j\cdot x}
\right\}_{j=-p}^{p}.
$$

```{code-cell} ipython3
:tags: [hide-input]

def dghelmholtz(fes, test_fes, omega, bndc):
    mesh = fes.mesh
    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    order = fes.globalorder

    alpha = order / (omega * h)
    beta = omega * h / order
    delta = omega * h / order

    u = fes.TrialFunction()
    v = test_fes.TestFunction()

    jump = lambda w: (w - w.Other()) * n
    avg_grad = lambda w: 0.5 * (grad(w) + grad(w.Other()))

    a = BilinearForm(fes, test_fes)
    a += grad(u) * grad(v) * dx - omega**2 * u * v * dx
    a += -(jump(u) * avg_grad(v) + avg_grad(u) * jump(v)) * dx(skeleton=True)
    a += 1j * omega * alpha * jump(u) * jump(v) * dx(skeleton=True)
    a += -beta / (1j * omega) * jump(grad(u)) * jump(grad(v)) * dx(skeleton=True)
    a += -delta * (u * grad(v) * n + grad(u) * n * v) * ds(skeleton=True)
    a += 1j * omega * (1 - delta) * u * v * ds(skeleton=True)
    a += -delta / (1j * omega) * (grad(u) * n) * (grad(v) * n) * ds(skeleton=True)

    f = LinearForm(test_fes)
    f += (1 - delta) * bndc * v * ds(skeleton=True)
    f += -delta / (1j * omega) * bndc * grad(v) * n * ds(skeleton=True)

    with TaskManager():
        a.Assemble()
        f.Assemble()
    return a, f
```

```{code-cell} ipython3
omega = 8
exact = exp(1j * omega * (1 / sqrt(2) * (x + y)))
grad_exact = CF((1j * omega / sqrt(2) * exact,1j * omega / sqrt(2) * exact))
n = specialcf.normal(2)
bndc = grad_exact * n + 1j * omega * exact

fes = trefftzfespace(
    mesh,
    order=5,
    eq="helmholtz",
    complex=True,
    dgjumps=True,
)
fes_test = trefftzfespace(
    mesh,
    order=5,
    eq="helmholtzconj",
    complex=True,
    dgjumps=True,
)

a, f = dghelmholtz(fes, fes_test, omega, bndc)
gfu = GridFunction(fes)

with TaskManager():
    gfu.vec.data = a.mat.Inverse() * f.vec

err = sqrt(Integrate((gfu - exact) * Conj(gfu - exact), mesh).real)
print(f"Helmholtz Trefftz dofs: {fes.ndof}")
print(f"Helmholtz L2 error:    {err:.3e}")

Draw(gfu, mesh, "gfu", animate_complex=True);
```

To inspect the available built-in spaces interactively:

```{code-cell} ipython3
trefftzfespace?
```
