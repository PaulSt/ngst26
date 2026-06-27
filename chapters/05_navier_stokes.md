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

# Embedded Trefftz for Navier-Stokes

For the stationary incompressible Navier-Stokes equations,

$$
-\nu\Delta u + (u\cdot\nabla)u + \nabla p = f,
\qquad
\nabla\cdot u = 0,
$$

the nonlinear part is the convective velocity in $(u\cdot\nabla)u$. The
embedded Trefftz construction enters after linearization: freeze one velocity
field and solve an Oseen problem in a Trefftz space adapted to that frozen
operator.

We use the following DG formulation

$$
\begin{aligned}
a_h^w((u,p),(v,q))
&=
\nu\int_{\Omega}\nabla u:\nabla v\,\mathrm{d}x
-
\int_{\Omega}
\left(
q\,\nabla\cdot u
+
p\,\nabla\cdot v
\right)
\,\mathrm{d}x
\\
&\quad
+
\int_{\Omega}
\left(
(w\cdot \nabla u)\cdot v
+
\frac12(\nabla\cdot w)\,u\cdot v
\right)
\,\mathrm{d}x
\\
&\quad
-
\int_{\mathcal F_h^{\mathrm{int}}}
\nu\{\!\{\nabla u\}\!\}n\cdot[\![ v]\!]
+
\nu\{\!\{\nabla v\}\!\}n\cdot[\![ u]\!]
-
\nu\frac{\alpha k^2}{h}
[\![ u]\!]\cdot[\![ v]\!]
\,\mathrm{d}s
\\
&\quad
-
\int_{\mathcal F_h^{\mathrm{int}}}
\{\!\{p\}\!\}\,n\cdot[\![ v]\!]
-
\{\!\{q\}\!\}\,n\cdot[\![ u]\!]
\,\mathrm{d}s
\\
&\quad
+
\int_{\mathcal F_h^{\mathrm{int}}}
(\{\!\{w\}\!\}\cdot n)
[\![ u]\!]\cdot\{\!\{v\}\!\}
+
\frac12([\![ w]\!]\cdot n)
\{\!\{u\cdot v\}\!\}
\,\mathrm{d}s
\\ &\quad +\text{ suitable boundary terms}.
\end{aligned}
$$

```{code-cell} ipython3
:tags: [hide-input]

from ngsolve import *
from netgen.occ import *
from ngsolve_book import Draw
from ngstrefftz import *
def VLap(u, dim):
    hesse = u.Operator("hesse")
    return CF(tuple(sum(hesse[i, j * (dim + 1)] for j in range(dim))for i in range(dim)))


def SolveOseen(mesh,k,nu,rhs=None,wn=None,trefftz=True,ubnd=None,bndname=".*",inletname=None,alpha=50):
    if hasattr(wn, "Operators") and "div" in wn.Operators():
        divwn = div(wn)
    elif wn is not None:
        if mesh.dim == 2:
            divwn = wn[0].Diff(x) + wn[1].Diff(y)
        elif mesh.dim == 3:
            divwn = wn[0].Diff(x) + wn[1].Diff(y) + wn[2].Diff(z)

    if inletname is None:
        inletname = bndname
    solver = "pardiso"
    stab = 1e-7

    V = VectorL2(mesh, order=k, dgjumps=True)
    Q = L2(mesh, order=k - 1, dgjumps=True)
    basefes = V * Q

    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size

    if trefftz:
        Vs = VectorL2(mesh, order=k - 2)
        Qs = L2(mesh, order=k - 1)
        test_fes = Vs * Qs
        u, p = basefes.TrialFunction()[0:2]
        wu, wp = test_fes.TestFunction()[0:2]
        oseen_residual = -nu * VLap(u, mesh.dim) + grad(p)
        if wn is not None:
            oseen_residual += grad(u) * wn
        op = oseen_residual * wu * dx
        op += div(u) * wp * dx

        emb = TrefftzEmbedding(op)
        fes = EmbeddedTrefftzFES(emb)
        if rhs is not None:
            upf = GridFunction(basefes)
            lop = rhs * wu * dx
            upf.vec.data = emb.GetParticularSolution(lop)
    else:
        fes = basefes

    u, v = fes.TrialFunction()[0], fes.TestFunction()[0]
    p, q = fes.TrialFunction()[1], fes.TestFunction()[1]

    def jump(v):
        return v - v.Other()

    def mean(v):
        return 0.5 * (v + v.Other())

    def mean2(u, v):
        return 0.5 * (u * v + u.Other() * v.Other())

    a = BilinearForm(fes)
    def dgterms(u,p,v,q):
        a = nu * InnerProduct(grad(u), grad(v)) * dx
        a += nu * alpha * k**2 / h * jump(u) * jump(v) * dx(skeleton=True)
        a += nu * (-mean(grad(u)) * n * jump(v) - mean(grad(v)) * n * jump(u)) * dx(skeleton=True)
        a += nu * alpha * k**2 / h * u * v * ds(skeleton=True, definedon=mesh.Boundaries(bndname))
        a += nu * (-grad(u) * n * v - grad(v) * n * u) * ds(skeleton=True, definedon=mesh.Boundaries(bndname))
        a += (mean(p) * n * jump(v) + mean(q) * n * jump(u)) * dx(skeleton=True)
        a += (p * v * n + q * u * n) * ds(skeleton=True, definedon=mesh.Boundaries(bndname))
        a += (-div(u) * q - div(v) * p) * dx
        a += -stab * p * q * dx
        if wn is not None:
            a += (grad(u) * wn) * v * dx
            a += -mean(wn) * n * jump(u) * mean(v) * dx(skeleton=True)
            a += 0.5 * divwn * u * v * dx
            a += -0.5 * jump(wn) * n * mean2(u, v) * dx(skeleton=True)
            a += -0.5 * wn * n * u * v * ds(skeleton=True,definedon=mesh.Boundaries(bndname),)
        return a
    a += dgterms(u,p,v,q)

    f = LinearForm(fes)
    if rhs is not None:
        f += rhs * v * dx
    if ubnd is not None:
        f += nu * alpha * k**2 / h * ubnd * v * ds(skeleton=True,definedon=mesh.Boundaries(inletname))
        f += nu * (-grad(v) * n * ubnd) * ds(skeleton=True,definedon=mesh.Boundaries(inletname))
        f += q * ubnd * n * ds(skeleton=True,definedon=mesh.Boundaries(inletname))
        if wn is not None:
            f += -0.5 * wn * n * ubnd * v * ds(skeleton=True,definedon=mesh.Boundaries(inletname))
    if trefftz and rhs is not None:
        mupf = GridFunction(basefes)
        mupf.vec.data = -1*upf.vec
        muf = mupf.components[0]
        mpf = mupf.components[1]
        f += dgterms(muf, mpf, v, q)


    a.Assemble()
    f.Assemble()

    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(inverse=solver) * f.vec

    if trefftz:
        polygfu = GridFunction(basefes)
        polygfu.vec.data = emb.Embed(gfu.vec)
        gfu = GridFunction(basefes)
        if rhs is not None:
            gfu.vec.data = polygfu.vec + upf.vec
        else:
            gfu = polygfu

    uh, ph = gfu.components[0:2]

    return uh, ph, fes.ndof
```

## Picard Linearization

Given a velocity $w$, the Oseen operator is

$$
\mathcal O_w(u,p)
=
-\nu\Delta u
+ (w\cdot\nabla)u
+ \nabla p,
\qquad
\nabla\cdot u=0.
$$

The Trefftz embedding uses as local operator.

Picard iteration freezes the convection field at the previous velocity:

$$
\begin{aligned}
\mathcal O_{u^{m}}(u^{m+1},p^{m+1}) &= f,\\
\nabla\cdot u^{m+1} &= 0.
\end{aligned}
$$

In each loop we solve the discrete problem with

$$
(u\cdot\nabla)u
\quad\leadsto\quad
(u^m\cdot\nabla)u^{m+1}.
$$

The embedded Trefftz space is updated in each iteration step.

:::{admonition} Picard loop
```python
wn = None
for step in range(maxiter):
    uh, ph = solve_oseen(mesh, nu, wn=wn, trefftz=True)
    if wn is not None and norm(uh - wn) < tol:
        break
    wn = uh
```

The next Oseen solve, and therefore the next embedded Trefftz space, is rebuilt for this new convection field.
:::

## Oseen Trefftz Space

On one element $K$, start with the discontinuous polynomial trial space

$$
V_h^k(K)
=
[\mathbb P^k(K)]^d \times \mathbb P^{k-1}(K)
$$

for velocity and pressure. The residual is tested in

$$
Q_h^k(K)
=
[\mathbb P^{k-2}(K)]^d \times \mathbb P^{k-1}(K).
$$

For fixed convection $w$, the embedded Oseen Trefftz space is

:::{admonition} 
$$
\mathbb T_{w,h}^k(K)
=
\left\{
(u_h,p_h)\in V_h^k(K):
\begin{array}{l}
\left(-\nu\Delta u + (w\cdot\nabla)u + \nabla p, v\right)_K=0\\
(\nabla\cdot u,q)_K =0
\end{array}
\quad\forall (v_h,q_h)\in Q_h^k(K)
\right\},
$$
:::

```{code-cell} ipython3
mesh = Mesh(unit_square.GenerateMesh(maxh=0.45))
k = 3
nu = 1.0
wn = CF((1, 0))

V = VectorL2(mesh, order=k, dgjumps=True)
Q = L2(mesh, order=k - 1, dgjumps=True)
base = V * Q

R = VectorL2(mesh, order=k - 2) * L2(mesh, order=k - 1)
u, p = base.TrialFunction()[0:2]
v, q = R.TestFunction()[0:2]

op = (-nu * VLap(u, mesh.dim) + grad(p) + grad(u) * wn) * v * dx
op += div(u) * q * dx

emb = TrefftzEmbedding(op)
tfes = EmbeddedTrefftzFES(emb)

print(f"full Oseen polynomial dofs: {base.ndof}")
print(f"embedded Oseen Trefftz dofs: {tfes.ndof}")
```

## Schäfer-Turek Benchmark

We now run the DFG 2D-1 Schäfer-Turek flow in the channel with a circular obstacle,

$$
\Omega=(0,2.2)\times(0,0.41)\setminus B_{0.05}(0.2,0.2).
$$

The inflow profile is parabolic,

$$
u_{\mathrm{in}}(y)
=
\left(
\frac{4U_{\max}y(0.41-y)}{0.41^2},
0
\right),
\qquad
U_{\max}=0.3,
\qquad
\nu=10^{-3}.
$$

```{code-cell} ipython3
:tags: [hide-input]

%%time
def solve_ns(mesh,k=4,nu=0.001,ubnd=None,bndname="inlet|wall|cyl",inletname="inlet",maxiter=100,tol=1e-8,alpha=50,trefftz=True):
    wn = CF((0,0))
    uh_old = wn

    for step in range(1, maxiter + 1):
        uh, ph, ndof = SolveOseen(mesh,k=k,nu=nu,wn=wn,trefftz=trefftz,ubnd=ubnd,bndname=bndname,inletname=inletname,alpha=alpha,)
        update = sqrt(Integrate(InnerProduct(uh - uh_old, uh - uh_old), mesh))
        print(f"Step: {step}, residual: {update}")
        if update < tol:
            return uh, ph, ndof, step

        uh_old = uh
        wn = uh

    return uh, ph, ndof, maxiter

ubnd = CF((4 * .3 * y * (0.41 - y) / (0.41 * 0.41), 0))
shape = Rectangle(2.2, 0.41).Circle(0.2, 0.2, 0.05).Reverse().Face()
shape.edges.name = "cyl"
shape.edges.Min(X).name = "inlet"
shape.edges.Max(X).name = "outlet"
shape.edges.Min(Y).name = shape.edges.Max(Y).name = "wall"
mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.05)).Curve(4)

with TaskManager():
    uh, ph, ndof, steps = solve_ns(mesh,k=4,ubnd=ubnd,trefftz=True)
```

```{code-cell} ipython3
print(f"embedded DG dofs: {ndof}")
print(f"Picard steps run: {steps}")

Draw(Norm(uh), mesh, "speed");
Draw(ph - Integrate(ph, mesh) / Integrate(1, mesh), mesh, "pressure");
```
