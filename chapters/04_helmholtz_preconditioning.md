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

# Helmholtz Preconditioning

The Helmholtz equation is the solver-facing example:

$$
-\Delta u - \omega^2 u = f.
$$

```{code-cell} ipython3
:tags: [hide-input]

from ngsolve import *
from netgen.occ import *
from ngsolve_book import Draw
from ngstrefftz import *
import matplotlib.pyplot as plt
SetNumThreads(4)
Lap = lambda u : sum(Trace(u.Operator('hesse')))
order = 6
omega = 320
maxh = 0.02
l = 2                    # number of circles in y-direction
dist = 0.3
radius = dist / 3
yoff = -dist * (l - 1) / 2
domain = MoveTo(-1, -1).Rectangle(2, 2).Face()
domain.edges.name = "transparent"
domain.edges.Min(X).name = "excitation"
for it in range(l):
    ycenter = yoff + dist * it
    hole = Circle((0, ycenter), radius).Face()
    hole.edges.name = "dirichlet"
    domain = domain - hole 
geo = OCCGeometry(domain, dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=maxh))
mesh.Curve(order)
```

For the embedded Trefftz space the local residual is

$$
(-\Delta u-\omega^2 u,q)_K=0
\qquad \forall q\in \mathbb P^{p-2}(K).
$$

The embedded Trefftz space stays polynomial!

```{code-cell} ipython3
basefes = L2(mesh, order=order, dgjumps=True,complex=True)
test_fes = L2(mesh, order=order-2, dgjumps=True,complex=True)
u,v = basefes.TnT()
w = test_fes.TestFunction()
op = (-Lap(u)-omega**2*u)*w*dx
with TaskManager():
    emb = TrefftzEmbedding(op)
    fes = EmbeddedTrefftzFES(emb)
```

## A scattering Problem

The live problem is deliberately small: a unit square with a circular obstacle,
a Gaussian incoming wave on the left boundary, impedance outer boundaries, and
a Dirichlet scatterer.

$$
\begin{aligned}
a_h(u,v)
={}&
\sum_{K\in\mathcal{T}_h}
\int_K
\left(
\nabla u\cdot\nabla v-\omega^2uv
\right)\,\mathrm{d}x
-
\int_{\mathcal{F}_h^{\mathrm{i}}}
\left(
\{\!\{\nabla u\}\!\}\cdot n\,
\left[\!\left[v\right]\!\right]
+
\{\!\{\nabla v\}\!\}\cdot n\,
\left[\!\left[u\right]\!\right]
\right)\,\mathrm{d}s
\\
&-
\mathrm{i}\omega
\int_{\mathcal{F}_h^{\mathrm{i}}}
\alpha\,
\left[\!\left[u\right]\!\right]
\left[\!\left[v\right]\!\right]
\,\mathrm{d}s
-
\frac{\mathrm{i}}{\omega}
\int_{\mathcal{F}_h^{\mathrm{i}}}
\beta\,
\left(
\left[\!\left[\nabla u\right]\!\right]\cdot n
\right)
\left(
\left[\!\left[\nabla v\right]\!\right]\cdot n
\right)
\,\mathrm{d}s
\\[0.3em]
&-
\int_{\Gamma_{\mathrm{o}}}
\delta\,\partial_nu\,v\,\mathrm{d}s
-
\int_{\Gamma_{\mathrm{o}}}
\delta\,\partial_nv\,u\,\mathrm{d}s
-
\mathrm{i}\omega
\int_{\Gamma_{\mathrm{o}}}
(1-\delta)\,uv\,\mathrm{d}s
-
\frac{\mathrm{i}}{\omega}
\int_{\Gamma_{\mathrm{o}}}
\delta\,\partial_nu\,\partial_nv\,\mathrm{d}s
\\[0.3em]
&-
\int_{\Gamma_{\mathrm{D}}}
\left(
\partial_nu\,v+\partial_nv\,u
\right)\,\mathrm{d}s
+
\int_{\Gamma_{\mathrm{D}}}
\alpha\,uv\,\mathrm{d}s .
\end{aligned}
$$

For the impedance condition
$
\partial_nu-\mathrm{i}\omega u=g
\quad\text{on the inlet }\Gamma_{\mathrm{i}},
$
the linear functional is

$$
f_h(v)
=
-\frac{\mathrm{i}}{\omega}
\int_{\Gamma_{\mathrm{i}}}
\delta\,g\,\partial_nv\,\mathrm{d}s
+
\int_{\Gamma_{\mathrm{i}}}
(1-\delta)\,g\,v\,\mathrm{d}s .
$$

```{code-cell} ipython3
:tags: [hide-input]

n = specialcf.normal(2)
h = specialcf.mesh_size

jmp = lambda u: u - u.Other()
avg = lambda u: 0.5 * (u + u.Other())

def DGHelmholtz(fes, omega, rhs=0, ibndc=0, ibnd=".*", obnd=".*", dbnd="",
                 alpha=None, beta=None, delta=None):
    order = fes.globalorder
    p = order/log(order+2)
    alpha = p/omega/h if alpha is None else alpha
    beta  = omega*h/p if beta is None else beta
    delta = omega*h/p if delta is None else delta

    u, v = fes.TnT()
    mesh = fes.mesh

    a = BilinearForm(fes)

    a += grad(u)*grad(v)*dx - omega**2*u*v*dx
    a += (-avg(grad(u))*n*jmp(v) -avg(grad(v))*n*jmp(u)) * dx(skeleton=True)
    a += -1j*omega*alpha*jmp(u)*jmp(v) * dx(skeleton=True)
    a += -1j/omega*beta * ( (jmp(grad(u))*n)*(jmp(grad(v))*n)) * dx(skeleton=True)

    # Impedance boundary: partial_n u - i omega u = ibndc
    a += -delta*grad(u)*n*v * ds(skeleton=True, definedon=mesh.Boundaries(obnd))
    a += -delta*grad(v)*n*u * ds(skeleton=True, definedon=mesh.Boundaries(obnd))
    a += -1j*omega*(1-delta)*u*v * ds(skeleton=True, definedon=mesh.Boundaries(obnd))
    a += -1j/omega*delta*(grad(u)*n)*(grad(v)*n) * ds(skeleton=True, definedon=mesh.Boundaries(obnd))

    # Dirichlet Nitsche terms
    a += (-grad(u)*n*v-grad(v)*n*u) * ds(skeleton=True, definedon=mesh.Boundaries(dbnd))
    a += alpha*u*v * ds(skeleton=True, definedon=mesh.Boundaries(dbnd))

    f = LinearForm(fes)
    f += -1j/omega*delta*ibndc*grad(v)*n * ds(skeleton=True, definedon=mesh.Boundaries(ibnd))
    f += (1-delta)*ibndc*v * ds(skeleton=True, definedon=mesh.Boundaries(ibnd))

    return a, f
```

## Domain-Decomposition Sweep

The sweep preconditioner partitions the mesh, builds local inverses on each subdomain, and applies multiplicative forward and backward passes. 

**Many thanks** to Michael & Joachim for this [example.](https://docu.ngsolve.org/ngs24/Helmholtz/HelmholtzIterative2D.html)

```{code-cell} ipython3
:tags: [hide-input]
def GenerateSubdomains(mesh, ndom):
    nbels = []
    for el in mesh.Elements(VOL):
        nbs = []
        for f in el.facets:
            for nb in mesh[f].elements:
                if nb != el:
                    nbs.append(nb.nr)
        nbels.append(nbs)
    import pymetis
    n_cuts, dddomains = pymetis.part_graph(ndom, adjacency=nbels)
    return n_cuts, dddomains

def GetDofLists(dddomains, fes):
    mesh = fes.mesh
    ndom = max(dddomains)+1
    domaindofs = [BitArray(fes.ndof) for i in range(ndom)]
    for domdof in domaindofs:
        domdof.Clear()
    for el in mesh.Elements(VOL):
        subdom = domaindofs[dddomains[el.nr]]
        dofis = fes.GetDofNrs(el)
        for d in dofis:
            if d>=0:
                subdom.Set(d)
    return domaindofs

ndomains = 52
print("nDomains:", ndomains)
ncuts, dddomains = GenerateSubdomains(mesh, ndomains)
gfdom = GridFunction(L2(mesh,order=0))
gfdom.vec.data = BaseVector(dddomains)
Draw (gfdom, mesh, "dddomains", settings={"Objects": {"Wireframe": False}, "Colormap": {"ncolors": ndomains}})
doflists = GetDofLists(dddomains, fes)
```

```{code-cell} ipython3
class DomainDecompositionPrecond(BaseMatrix):
    def __init__(self, amat, doflists, sweeps=1):
        super().__init__()
        self.blockinv = [
            amat.Inverse(freedofs, inverse="sparsecholesky")
            for freedofs in doflists
        ]
        self.sweeps = sweeps
        
    def Sweep(self, rhs, sol):
        for _ in range(self.sweeps):
            for inv in self.blockinv[::-1]:
                inv.Smooth(sol, rhs)
            for inv in self.blockinv:
                inv.Smooth(sol, rhs)

    def Mult(self, x, y):
        y[:] = 0
        self.Sweep(x, y)
```

The same object can be used as a stationary iteration or as a right
preconditioner inside GMRES.

```{code-cell} ipython3
incoming = -1e1*1j*exp(-2e1*(y**2))
a,f = DGHelmholtz(fes, omega, ibndc=incoming, ibnd="excitation", obnd="excitation|transparent", dbnd="dirichlet")
from ngsolve.krylovspace import *
with TaskManager():
    a.Assemble()
    f.Assemble()
    precond = DomainDecompositionPrecond(a.mat,doflists)
    inv = CGSolver(mat=a.mat, pre=precond, tol=1e-5, printrates=True)
    gfu = GridFunction(fes)
    gfu.vec.data = inv * f.vec
```

```{code-cell} ipython3
polygfu = emb.Embed(gfu)
Draw(polygfu, mesh, "sol", min=-0.01, max=0.01, order=order, animate_complex=True, deformation=True, scale=1,euler_angles=[-70,0.4,2], settings={"Objects": {"Wireframe": False}});
```

## Is it sweeping?

This is essentially a upper-lower block Gauss-Seidel preconditioner.
However, due to the special jump-jump terms in our bilinear form, the subdomain solver acts as a good sweeping method

```{code-cell} ipython3
shape = MoveTo(-1, -1).Rectangle(2, 2).Face()
shape.edges.name = "obnd"
shape.edges.Min(X).name = "ibnd"
mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.1))
```

Compare the results to the simpler jump-jump choice of beta=0

```{code-cell} ipython3
:tags: [hide-input]

omega = 20
basefes = L2(mesh, order=order, dgjumps=True,complex=True)
test_fes = L2(mesh, order=order-2, dgjumps=True,complex=True)
u,v = basefes.TnT()
w = test_fes.TestFunction()
op = (-Lap(u)-omega**2*u)*w*dx
with TaskManager():
    emb = TrefftzEmbedding(op)
    fes = EmbeddedTrefftzFES(emb)
    
a,f = DGHelmholtz(fes, omega, beta=0, ibnd="ibnd", obnd=".*", ibndc=incoming)
with TaskManager():
    a.Assemble()
    f.Assemble()
    gfu = GridFunction(fes)
    sweep = DomainDecompositionPrecond(a.mat, GetDofLists(GenerateSubdomains(mesh, 4)[1], fes))
    sweep.Sweep(f.vec, gfu.vec)
    Draw(emb.Embed(gfu))
```

against $\beta> 0$. The beta term is crucial, it creates impedence boundary conditions between the subdomains.

```{code-cell} ipython3
:tags: [hide-input]

a,f = DGHelmholtz(fes, omega, ibnd="ibnd", obnd=".*", ibndc=incoming)
with TaskManager():
    a.Assemble()
    f.Assemble()
    gfu = GridFunction(fes)
    sweep = DomainDecompositionPrecond(a.mat, GetDofLists(GenerateSubdomains(mesh, 4)[1], fes))
    sweep.Sweep(f.vec, gfu.vec)
    Draw(emb.Embed(gfu))

    sweep.sweeps = 50
    sweep.Sweep(f.vec, gfu.vec)
    Draw(emb.Embed(gfu))
```

## How good is the sweeper?
We take a look at a trapping geometry!

```{code-cell} ipython3
:tags: [hide-input]

import math
cx, cy = 0.66, 0.50
outer_radius = 0.22
inner_radius = 0.12
slot_height = 0.12
outer_radius = 0.25
inner_radius = 0.2
slot_height = 0.1
slot_x0 = cx - outer_radius - 0.02
slot_width = outer_radius - inner_radius + 0.05

outer_circle = WorkPlane().Circle(cx, cy, outer_radius).Face()
inner_circle = WorkPlane().Circle(cx, cy, inner_radius).Face()
slot = MoveTo(slot_x0, cy - slot_height / 2).Rectangle(slot_width,slot_height).Face()
cavity_wall = outer_circle - inner_circle - slot
shape = Rectangle(1, 1).Face() - cavity_wall
shape.edges.name = "dirichlet"
shape.edges.Min(X).name = "ibnd"
shape.edges.Max(X).name = "obnd"
shape.edges.Min(Y).name = "obnd"
shape.edges.Max(Y).name = "obnd"
dbnd = "dirichlet"
elements_per_wavelength = 6
omega = 50
maxh = 2.0 * math.pi / (omega * elements_per_wavelength)
mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
fes = L2(mesh, order=order, dgjumps=True,complex=True)
a,f = DGHelmholtz(fes, omega, ibnd="ibnd", obnd=".*", dbnd="dirichlet", ibndc= -10.0j * exp(-20.0 * (2.0 * y - 1.0) ** 2))
with TaskManager():
    a.Assemble()
    f.Assemble()
    gfu = GridFunction(fes)
    sweep = DomainDecompositionPrecond(a.mat, GetDofLists(GenerateSubdomains(mesh, 16)[1], fes))
    residuals2 = []
    for i in range(50):
        sweep.Sweep(f.vec, gfu.vec)
        residuals2.append(Norm(a.mat * gfu.vec - f.vec))
    Draw(gfu)
```

embedded Trefftz DG sweep:

```{code-cell} ipython3
:tags: [hide-input]
basefes = L2(mesh, order=order, dgjumps=True,complex=True)
test_fes = L2(mesh, order=order-2, dgjumps=True,complex=True)
u,v = basefes.TnT()
w = test_fes.TestFunction()
op = (-Lap(u)-omega**2*u)*w*dx
with TaskManager():
    emb = TrefftzEmbedding(op)
    fes = EmbeddedTrefftzFES(emb)
a,f = DGHelmholtz(fes, omega, ibnd="ibnd", obnd=".*", dbnd="dirichlet", ibndc= -10.0j * exp(-20.0 * (2.0 * y - 1.0) ** 2))
with TaskManager():
    localpre = Preconditioner(a, "local")
    a.Assemble()
    f.Assemble()
    gfu = GridFunction(fes)
    sweep = DomainDecompositionPrecond(a.mat, GetDofLists(GenerateSubdomains(mesh, 16)[1], fes))
    residuals = []
    for i in range(50):
        sweep.Sweep(f.vec, gfu.vec)
        residuals.append(Norm(a.mat * gfu.vec - f.vec))
    Draw(emb.Embed(gfu))
    gfu.vec.data = a.mat.Inverse()*f.vec
    Draw(emb.Embed(gfu))
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt
plt.semilogy(range(1, len(residuals) + 1), residuals, label="Trefftz sweep")
plt.semilogy(range(1, len(residuals2) + 1), residuals2, label="DG")
plt.xlabel("Sweeps")
plt.ylabel("Residual")
plt.legend()
plt.show()
```
