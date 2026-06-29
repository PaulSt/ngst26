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

# Anisotropic Elements

The anisotropic example is a singularly perturbed reaction-diffusion problem,

$$
-\varepsilon^2\Delta u + u = f,
$$

where boundary layers make stretched elements useful. 
The embedded Trefftz point is that the local residual tests do not need to be the full tensor-product trial space. 
We keep the DG trial space, but test the residual with a compact anisotropic tensor-product space.

$$
\begin{array}{l|rrrr|rrrr}
\hline
&
\textbf{2D} & & &
&
\textbf{3D} & & & 
\\
\text{method}\downarrow\;\backslash\;p\rightarrow
& 2 & 3 & 4 & 5
& 2 & 3 & 4 & 5
\\
\hline
\texttt{ncdof}_{\mathrm{DG}}/N_{\mathrm{El}}
& 9 & 16 & 25 & 36
& 27 & 64 & 125 & 216
\\
\texttt{ncdof}_{\mathrm{TDG}}/N_{\mathrm{El}}
& 5 & 7 & 9 & 11
& 9 & 16 & 25 & 36
\\
\texttt{ncdof}_{\mathrm{HDG}}/N_{\mathrm{El}}
& 6 & 8 & 10 & 12
& 27 & 48 & 75 & 108
\\
\hline
\texttt{nnze}_{\mathrm{DG}}/N_{\mathrm{El}}
& 405 & 1280 & 3125 & 6480
& 5103 & 28672 & 109375 & 326592
\\
\texttt{nnze}_{\mathrm{TDG}}/N_{\mathrm{El}}
& 125 & 245 & 405 & 605
& 567 & 1792 & 4375 & 9072
\\
\texttt{nnze}_{\mathrm{HDG}}/N_{\mathrm{El}}
& 126 & 224 & 350 & 504
& 2673 & 8448 & 20625 & 42768
\\
\hline
\end{array}
$$


## Stability Picture

The analysis asks for two independent stability mechanisms: $a_h$ must be
well-posed on $\mathbb T_h$, and the local maps
$\mathcal L_K:\mathbb L_h(K)\to Q_h(K)'$ must control the lifting component,

$$
\|\mathcal L_K u_h\|_{Q_h(K)'}^2
\gtrsim
\|u_h\|_{V_h(K)}^2
\qquad \forall u_h\in\mathbb L_h(K).
$$

For variable coefficients and anisotropic test spaces, the analysis often starts with a prototype operator $\mathcal L_{K,0}$ and treats the actual operator as a perturbation. 
On the lifting space, assume

$$
\|\mathcal L_K u-\mathcal L_{K,0}u\|_{Q_h(K)'}
\leq
\gamma\|\mathcal L_{K,0}u\|_{Q_h(K)'},
\qquad 0<\gamma<1 .
$$

Then the local lifting solve remains stable. Geometrically, the Trefftz space rotates inside the polynomial space, but it does not collide with the chosen lifting complement.


:::{figure} _static/tikz/prototype-perturbation.svg
:width: 50%
:::

For Laplace, $V_h^p(K)=\mathbb P^p(K)$ and $Q_h(K)=\mathbb P^{p-2}(K)$ recover harmonic polynomials because $\Delta\mathbb P^p(K)\subset \mathbb P^{p-2}(K)$. 


## The TP0 Constraint Space

On the element $\widetilde K=\widetilde K_1\times\widetilde K_2$, the TP0 tests vanish on the two long edges

$$
Q_h(\widetilde K)
=
\{\widetilde q_h\in \mathbb P^p_\otimes(\widetilde K):
\widetilde q_h|_{\partial\widetilde K_1\times \widetilde K_2}=0\}.
$$

That choice gives $(p+1)(p-1)$ local constraints and leaves $2p+2$ embedded Trefftz unknowns per quadrilateral.

```{code-cell} ipython3
:tags: [hide-input]

import math

import matplotlib.pyplot as plt
import netgen.meshing as ngm
from netgen.geom2d import SplineGeometry
from ngsolve import *
from ngsolve import TensorProductTools
from ngsolve_book import Draw
from ngstrefftz import *
SetNumThreads(4)

Lap = lambda u: sum(Trace(u.Operator("hesse")))


def make_tensor_product_mesh(mesh1, mesh2):
    ngmesh = TensorProductTools.MakeTensorProductMesh(mesh1, mesh2)
    for el in ngmesh.Elements1D():
        if el.index < 1:
            el.index = 1
    return ngmesh


def graded_line_mesh(n, sigma):
    ngmesh = ngm.Mesh(dim=1)
    pids = []

    c = 1 - math.exp(-1 / sigma)
    left = [-1 - sigma * math.log(1 - c * (i + 1) / (n + 1)) for i in range(n)]
    right = [1 + sigma * math.log(1 - c * (i + 1) / (n + 1)) for i in range(n - 1, -1, -1)]
    points = [-1] + left + right + [1]

    for xi in points:
        pids.append(ngmesh.Add(ngm.MeshPoint(ngm.Pnt(xi, 0, 0))))

    TensorProductTools.AddEdgeEls(0, 1, 1, len(points) - 1, pids, ngmesh)
    ngmesh.Add(ngm.Element0D(pids[0], index=1))
    ngmesh.Add(ngm.Element0D(pids[-1], index=2))
    ngmesh.SetBCName(0, "left")
    ngmesh.SetBCName(1, "right")
    return ngmesh
```

We can see the constraint space directly on a graded tensor-product mesh.

```{code-cell} ipython3
meshx = Mesh(graded_line_mesh(1, sigma=0.2))
mesh = Mesh(make_tensor_product_mesh(meshx, meshx))
tp0_fes = TP0FESpace(mesh, order=2)
up = GridFunction(tp0_fes, multidim=tp0_fes.ndof)

for i, vec in enumerate(up.vecs):
    vec.data[:] = 0
    vec.data[i] = 1

Draw(up,mesh,"basis",animate=True,interpolate_multidim=False,min=-1, max=1, deformation=True, scale=.3, euler_angles=[-70,0.4,2],)
```

## Example 1 Recap

The square example uses the exact solution

$$
u(x,y)=\left(1-\frac{\cosh(x/\varepsilon)}{\cosh(1/\varepsilon)}\right)\left(1-\frac{\cosh(y/\varepsilon)}{\cosh(1/\varepsilon)}\right),
$$

so the layers sit on the four sides of $(-1,1)^2$.

```{code-cell} ipython3
:tags: [hide-input]

order = 5


eps = 4.0e-2
cosh = lambda z: (exp(z) + exp(-z)) / 2
exact = (1 - cosh(x / eps) / cosh(1 / eps)) * (
    1 - cosh(y / eps) / cosh(1 / eps)
)
rhs = -eps**2 * (exact.Diff(x).Diff(x) + exact.Diff(y).Diff(y)) + exact
sigma = (order+0.5)*eps * (3/exp(1))

meshx = Mesh(graded_line_mesh(10,sigma))
mesh = Mesh(make_tensor_product_mesh(meshx, meshx))
```

$$
\begin{aligned}
a_h(u,v)
&=
\int_{\Omega} \varepsilon^2\nabla u \cdot \nabla v \,\mathrm{d}x
+
\int_{\Omega}  u v \,\mathrm{d}x
\\ &\quad
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

def reaction_diffusion_form(u, v, eps, alpha, alpha_bnd):
    n = specialcf.normal(2)
    h = AdjacentFaceSizeCF()
    jump = lambda z: z - z.Other()
    avg = lambda z: h / (h + h.Other()) * z + h.Other() / (h + h.Other()) * z.Other()

    form = eps**2 * grad(u) * grad(v) * dx + u * v * dx
    form += eps**2 * (-avg(grad(u)) * n * jump(v)- avg(grad(v)) * n * jump(u)) * dx(skeleton=True)
    form += alpha * jump(u) * jump(v) * dx(skeleton=True)
    form += alpha_bnd * u * v * ds(skeleton=True)
    form += eps**2 * (-n * grad(u) * v - n * grad(v) * u) * ds(skeleton=True)
    return form


def reaction_diffusion_rhs(rhs, v, eps, alpha_bnd, bndc=0):
    n = specialcf.normal(2)
    form = rhs * v * dx
    form += alpha_bnd * bndc * v * ds(skeleton=True)
    form += eps**2 * (-n * grad(v) * bndc) * ds(skeleton=True)
    return form


def solve_embedded_from_embedding(basefes, tfes, emb, order, eps, rhs):
    h = AdjacentFaceSizeCF()
    alpha = 10 * eps**2 * order**2 / (h + h.Other())
    alpha_bnd = 10 * eps**2 * order**2 / h

    gfu = GridFunction(basefes)
    with TaskManager():
        gfu.vec.data = emb.GetParticularSolution()

    ut, vt = tfes.TnT()
    a = BilinearForm(tfes)
    a += reaction_diffusion_form(ut, vt, eps, alpha, alpha_bnd)

    f = LinearForm(tfes)
    f += reaction_diffusion_rhs(rhs, vt, eps, alpha_bnd)
    minus_gfu = GridFunction(basefes)
    minus_gfu.vec.data = -gfu.vec
    f += reaction_diffusion_form(minus_gfu, vt, eps, alpha, alpha_bnd)

    with TaskManager():
        a.Assemble()
        f.Assemble()
        sol_t = a.mat.Inverse(inverse="sparsecholesky") * f.vec

    gfu.vec.data += emb.Embed(sol_t)
    return gfu
```

The only code to remember is the embedded Trefftz space construction:

```{code-cell} ipython3
basefes = L2(mesh, order=order, dgjumps=True)
testfes = TP0FESpace(mesh, order=order, allow_both_axes_zero=False)

u = basefes.TrialFunction()
w = testfes.TestFunction()

op = (-eps**2 * Lap(u) + u) * w * dx
lop = rhs * w * dx
emb = TrefftzEmbedding(op, lop)
tfes = EmbeddedTrefftzFES(emb)

gfu = solve_embedded_from_embedding(basefes, tfes, emb, order, eps, rhs)

print(f"error: {sqrt(Integrate((gfu - exact) ** 2, mesh)):.3e}")

Draw(gfu, mesh, "Example 1: embedded Trefftz boundary layer");
```

## Curved Anisotropic Elements

For curved quadrilaterals, the paper maps the reference square first to the
stretched pullback rectangle and then to the physical element:

$$
\widehat K
\xrightarrow{\ F_K\ }
\widetilde K
\xrightarrow{\ G_K\ }
K.
$$

:::{figure} _static/tikz/anisotropic-element-map.svg
:name: fig-anisotropic-element-map
:alt: TikZ diagram mapping a reference square to a stretched rectangle and then to a curved physical element.
:width: 86%

The TikZ picture shows the reference-to-physical mapping. The TP0 space is
defined on $\widetilde K$ and pushed forward by $G_K$.
:::

## Example 2: Circular Boundary Layer

The circular experiment uses the unit disk and

$$
u(r)=1-\frac{\cosh(r/\varepsilon)}{\cosh(1/\varepsilon)},
\qquad
\varepsilon^2=10^{-5}.
$$

The central disk is triangulated. The outer annulus is built from curved
quadrilateral layers, graded toward the boundary layer at $r=1$.

```{code-cell} ipython3
:tags: [hide-input]

def inv_polar(vx, vy):
    radius = math.sqrt(vx**2 + vy**2)
    angle = math.atan2(vy, vx)
    if angle < 0:
        angle += 2 * math.pi
    return radius, angle


def hybrid_circle_mesh(r_layer, h_inner, bnd="dirichlet", order_geom=5, deformstart=0):
    mapping = lambda theta, radius: (
        radius * math.cos(theta),
        radius * math.sin(theta),
    )
    mesh_nodes = {}

    ngmesh = ngm.Mesh()
    geo = SplineGeometry()
    geo.AddCircle((0, 0), r_layer[-1], leftdomain=1, rightdomain=0, bc=bnd)
    ngmesh.SetGeometry(geo)
    ngmesh.dim = 2

    geo_inner = SplineGeometry()
    geo_inner.AddCircle((0, 0), r_layer[1], leftdomain=1, rightdomain=0, bc="inner")
    mesh_inner = geo_inner.GenerateMesh(maxh=h_inner)

    pmap = {}
    for el in mesh_inner.Elements2D():
        for vertex in el.vertices:
            if vertex not in pmap:
                pmap[vertex] = ngmesh.Add(mesh_inner[vertex])
                point = mesh_inner[vertex].p
                mesh_nodes[pmap[vertex].nr] = (point[0], point[1])

    idx_dom = ngmesh.AddRegion("inner", dim=2)
    for el in mesh_inner.Elements2D():
        curved_el = ngm.Element2D(idx_dom, [pmap[vertex] for vertex in el.vertices])
        curved_el.curved = True
        ngmesh.Add(curved_el)

    pids_inner = []
    for edge in mesh_inner.Elements1D():
        for vertex in edge.vertices:
            point_id = pmap[vertex]
            if point_id not in pids_inner:
                pids_inner.append(point_id)

    pids_inner.sort(
        key=lambda point_id: inv_polar(
            mesh_nodes[point_id.nr][0],
            mesh_nodes[point_id.nr][1],
        )[1]
    )
    pids_inner.append(pids_inner[0])

    def make_layer(pids_start, r_end):
        nx = len(pids_start) - 1
        angles = []
        zero_count = 0
        for point_id in pids_start:
            node_coord = mesh_nodes[point_id.nr]
            angle = inv_polar(node_coord[0], node_coord[1])[1]
            if abs(angle) < 1e-13:
                angles.append(0 if zero_count == 0 else 2 * math.pi)
                zero_count += 1
            else:
                angles.append(angle)

        pids_end = []
        for j in range(nx + 1):
            x_end, y_end = mapping(angles[j], r_end)
            if j == nx:
                pids_end.append(pids_end[0])
            else:
                node_id = ngmesh.Add(ngm.MeshPoint(ngm.Pnt(x_end, y_end, 0)))
                pids_end.append(node_id)
                mesh_nodes[node_id.nr] = (x_end, y_end)

        for i in range(nx):
            elpids = [pids_start[i + 1], pids_start[i], pids_end[i], pids_end[i + 1]]
            el = ngm.Element2D(idx_dom, elpids)
            el.curved = True
            ngmesh.Add(el)
        return pids_end

    pids_next = pids_inner
    for layer_index in range(len(r_layer) - 2):
        pids_next = make_layer(pids_next, r_layer[layer_index + 2])

    for i in range(len(pids_next) - 1):
        ngmesh.Add(ngm.Element1D([pids_next[i], pids_next[i + 1]], index=1))
    ngmesh.SetBCName(0, bnd)
    ngmesh.Compress()

    mesh = Mesh(ngmesh)

    fes_p1 = H1(mesh, complex=False, order=1, dirichlet=[])
    r_h = GridFunction(fes_p1)
    r_h.vec.data[:] = 0.0
    for coord_nr, coord in mesh_nodes.items():
        r_h.vec.data[coord_nr - 1] = math.sqrt(coord[0] ** 2 + coord[1] ** 2)

    r_hrelax = GridFunction(fes_p1)
    r_hrelax.vec.data[:] = 0.0
    for coord_nr, coord in mesh_nodes.items():
        if math.sqrt(coord[0] ** 2 + coord[1] ** 2) >= r_layer[deformstart] - 1e-3:
            r_hrelax.vec.data[coord_nr - 1] = math.sqrt(coord[0] ** 2 + coord[1] ** 2)

    r_dist = sqrt(x**2 + y**2)
    theta_perfect = r_hrelax * CoefficientFunction(
        ((r_h - r_dist) * x / r_dist, (r_h - r_dist) * y / r_dist)
    )

    fes_high_order = VectorH1(mesh, order=order_geom)
    theta_h = GridFunction(fes_high_order)
    theta_h.Set(theta_perfect)
    return mesh, theta_h


def circular_mesh(order, hnr, maxh_case):
    sigma = (order + 0.5) * eps_circle * (3 / math.e)
    c = 1 - math.exp(-1 / sigma)
    graded = [-sigma * math.log(1 - c * (i + 1) / (hnr + 1)) for i in range(hnr)]

    if maxh_case == 0:
        maxh = 0.15
        layers = [0.5, 0.62, 0.74, 0.87]
    else:
        maxh = 0.09
        layers = [0.5, 0.57, 0.64, 0.71, 0.8, 0.9]

    radii = [0.0] + layers + [1 - value for value in reversed(graded)] + [1.0]
    mesh, theta_h = hybrid_circle_mesh(radii, maxh, order_geom=order, deformstart=2)
    mesh.SetDeformation(theta_h)
    return mesh, theta_h
```

```{code-cell} ipython3
:tags: [hide-input]

eps_circle = 10 ** (-5 / 2)
exact_circle = 1 - cosh(sqrt(x**2 + y**2) / eps_circle) / cosh(1 / eps_circle)
rhs_circle = -eps_circle**2 * (
    exact_circle.Diff(x).Diff(x) + exact_circle.Diff(y).Diff(y)
) + exact_circle


def solve_reaction_diffusion(mesh, order, eps, rhs, method):
    fes = L2(mesh, order=order, dgjumps=True)
    h = AdjacentFaceSizeCF()
    alpha = 10 * eps**2 * order**2 / (h + h.Other())
    alpha_bnd = 10 * eps**2 * order**2 / h

    if method == "dg":
        gfu = GridFunction(fes)
        u, v = fes.TnT()
        a = BilinearForm(fes)
        a += reaction_diffusion_form(u, v, eps, alpha, alpha_bnd)
        f = LinearForm(fes)
        f += reaction_diffusion_rhs(rhs, v, eps, alpha_bnd)
        with TaskManager():
            a.Assemble()
            f.Assemble()
            gfu.vec.data = a.mat.Inverse(inverse="sparsecholesky") * f.vec
        return gfu, fes

    basefes = L2(mesh, order=order, dgjumps=True)
    testfes = TP0FESpace(mesh, order=order, allow_both_axes_zero=False)
    u = basefes.TrialFunction()
    w = testfes.TestFunction()
    op = (-eps**2 * Lap(u) + u) * w * dx
    lop = rhs * w * dx
    emb = TrefftzEmbedding(op, lop)
    tfes = EmbeddedTrefftzFES(emb)
    gfu = solve_embedded_from_embedding(basefes, tfes, emb, order, eps, rhs)
    return gfu, tfes
```

```{code-cell} ipython3
:tags: [hide-input]

preview_mesh, _ = circular_mesh(order=5, hnr=6, maxh_case=1)
preview_gfu, preview_fes = solve_reaction_diffusion(
    preview_mesh,
    order=5,
    eps=eps_circle,
    rhs=rhs_circle,
    method="embt",
)

print(f" error: {sqrt(Integrate((gfu - exact) ** 2, mesh)):.3e}")

Draw(preview_gfu, preview_mesh, "Example 2: embedded Trefftz on circular mesh");
```

The convergence loop uses the fine initial circular mesh, polynomial orders
$p=4,5$, DG and embedded Trefftz, and eight radial layer refinements. We measure
the plain $L^2$ error.

```{code-cell} ipython3
:tags: [hide-input]

def run_circular_convergence(order_values=(4, 5), hnr_values=range(1, 9)):
    rows = []
    for order in order_values:
        for hnr in hnr_values:
            mesh_case, _ = circular_mesh(order, hnr, maxh_case=1)

            for method in ("embt", "dg"):
                gfu, fes = solve_reaction_diffusion(mesh_case,order=order,eps=eps_circle,rhs=rhs_circle,method=method,)
                l2error = sqrt(Integrate((gfu - exact_circle) ** 2, mesh_case))
                rows.append({"method": method,"order": order,"hnr": hnr,"ndof": fes.ndof,"l2error": float(l2error),})
    return rows

circular_results = run_circular_convergence()
```

```{code-cell} ipython3
:tags: [hide-input]

styles = {
    "dg": {"color": "#2a6fbb", "marker": "o", "label": "DG"},
    "embt": {"color": "#d95f02", "marker": "s", "label": "Embedded"},
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
for order, ax in zip((4, 5), axes):
    for method in ("dg", "embt"):
        points = [
            row
            for row in circular_results
            if row["method"] == method and row["order"] == order
        ]
        points.sort(key=lambda row: row["ndof"])
        style = styles[method]
        ax.loglog(
            [row["ndof"] for row in points],
            [row["l2error"] for row in points],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.0,
            markersize=5,
            label=style["label"],
        )

    ax.set_title(rf"fine initial mesh, $p={order}$")
    ax.set_xlabel("degrees of freedom")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)

axes[0].set_ylabel(r"$\|u-u_h\|_{L^2(\Omega)}$")
axes[1].legend(loc="upper right", fontsize=9)
fig.suptitle("Example: circular boundary-layer convergence")
plt.show()
```
