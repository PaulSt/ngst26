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

# Can we enforce global coupling?

We augment the Embedded Trefftz Method:

- underyling FE-space: $\mathbb{V}_h$
- constraint FE-space: $\mathbb{Z}_h$
- Trefftz test space: $\mathbb{Q}_h$

\begin{align}
\mathbb{T}_{h,f} := \{ v_h \in \mathbb{V}_h \mid 
\exists  y_h \in \mathbb{Z}_h, 
 \text{ s.t. } \forall K \in \mathcal{T}_h: &~ 
 (\mathcal{L}_K v_h,q_h)_K = f(q_h) \quad \forall q_h \in \mathbb{Q}_h,\\ 
 &~ \mathcal{C}_K (v_h, z_h) = \mathcal{D}_K (y_h, z_h) \quad \forall z_h \in \mathbb{Z}_h
 \}
\end{align}

## ngstSpaceKit

Let's start with the extreme case:

$L \equiv 0, f \equiv 0$
\begin{align}
\mathbb{T}_{h,f} := \{ v_h \in \mathbb{V}_h \mid 
\exists  y_h \in \mathbb{Z}_h, 
 \text{ s.t. } \forall K \in \mathcal{T}_h: &~
 \mathcal{C}_K (v_h, z_h) = \mathcal{D}_K (y_h, z_h) \quad \forall z_h \in \mathbb{Z}_h
 \}
\end{align}

To compute the element-wise embedding matrix of the constraints we compute on each element

$$
\begin{align*}
& C = \mathcal{C}_K (v_h, z_h)\qquad D=\mathcal{D}_K (y_h, z_h)\\
& T = C^\dagger D
\end{align*}
$$

Several applications and examples are available in [ngstSpaceKit](https://johann-cm.codeberg.page/ngstspacekit)

```{code-cell} ipython3
:tags: [hide-input]

from ngsolve import *#L2, H1, Mesh, unit_square, dx, BND, BBND, GridFunction, FacetFESpace, ElementId, VOL, FESpace, NormalFacetFESpace, specialcf, CoefficientFunction, grad
from ngstrefftz import TrefftzEmbedding, EmbeddedTrefftzFES
#from ngstSpaceKit import bubbles
#from ngstSpaceKit.diffops import del_x, del_y, del_xx, del_xy, del_yy, hesse
from ngsolve_book import Draw

Lap = lambda u: sum(Trace(u.Operator("hesse")))

mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))

def EdgeBubble(mesh: Mesh, order: int, dirichlet: str = "") -> FESpace:
    """
    The `EdgeBubble` space only contains the edge bubble basis functions
    of the standard `ngsolve.comp.H1` space with order `order`.

    # Raises
    `ValueError`, if `order < 2`. There are no edge bubbles of order 1 or less.
    """
    if order < 2:
        raise ValueError("order must be >= 2")

    bubble = H1(mesh, order=order, dirichlet=dirichlet)
    for dofNr in range(bubble.ndof):
        bubble.SetCouplingType(dofNr, COUPLING_TYPE.UNUSED_DOF)

    for edge in mesh.edges:
        for dof in bubble.GetDofNrs(edge):
            bubble.SetCouplingType(dof, COUPLING_TYPE.INTERFACE_DOF)
    return Compress(bubble)

def del_x(f: CoefficientFunction) -> CoefficientFunction:
    """
    partial derivative in first coordinate direction
    """
    return grad(f)[0]


def del_y(f: CoefficientFunction) -> CoefficientFunction:
    """
    partial derivative in second coordinate direction
    """
    return grad(f)[1]


def del_z(f: CoefficientFunction) -> CoefficientFunction:
    """
    partial derivative in third coordinate direction
    """
    return grad(f)[2]


def del_xx(f: CoefficientFunction) -> CoefficientFunction:
    """
    partial derivative of second order, in first and first coordinate direction
    """
    return hesse(f)[0, 0]


def del_xy(f: CoefficientFunction) -> CoefficientFunction:
    """
    partial derivative of second order, in first and second coordinate direction
    """
    return hesse(f)[0, 1]


def del_yy(f: CoefficientFunction) -> CoefficientFunction:
    """
    partial derivative of second order, in second and second coordinate direction
    """
    return hesse(f)[1, 1]

def hesse(f: CoefficientFunction) -> CoefficientFunction:
    """
    Hesse matrix of a function.
    """
    return f.Operator("hesse")


def laplace(f: GridFunction) -> CoefficientFunction:
    """
    Laplace operator on a function that is scalar or vector valued.
    """
    f_hesse = hesse(f)
    dim = f.space.mesh.dim
    if len(f.shape) == 0:
        # f is scalar valued
        return Trace(f_hesse)
    elif len(f.shape) == 1:
        # f is vector valued
        return CF(
            tuple(
                # f_hesse[j,:] == f[j].Operator("hesse")
                # it would be nicer to sum over all diagonal indices f_hesse[j, i, i] for i in range (dim),
                # but ngsolve ravels dim. 2 and 3 into one dimension, so
                # (i,i) -> i*dim + i
                sum(f_hesse[j, i] for i in range(0, dim * dim, dim + 1))
                for j in range(dim)
            )
        )



def draw_basis(space: FESpace, mini=-0.5, maxi=0.5, height=0.5, vectorial=False) -> None:
    gfshow = GridFunction(space, multidim=0)
    gf = GridFunction(space)
    dofs = space.GetDofNrs(ElementId(VOL, 5))
    for dof in dofs:
        if dof < 0:
            continue
        gf.vec[:] = 0
        gf.vec[dof] = height
        gfshow.AddMultiDimComponent(gf.vec)
    Draw (gfshow, mesh, deformation=not vectorial, interpolate_multidim=False, animate=True, autoscale=False, min=mini, max=maxi, settings={"subdivision":20, "colorbar":False, "vectors": vectorial}, euler_angles=[-60, 5, 30], order=space.globalorder ,  vectors={"grid_size" : 20, "offset" : 0.5 })

import scipy.sparse as sp
import matplotlib.pyplot as plt

def plot_spy(fes, markersize=None):
    PP = fes.GetEmbedding().GetEmbedding()
    rows, cols, data = PP.COO()

    PPsp = sp.coo_matrix((data, (cols, rows)), shape=(PP.width, PP.height))
    plt.figure(figsize=(15, 4.5)); plt.spy(PPsp, markersize=markersize); plt.show()
```

### Crouzeix-Raviart
- $P^k$ space, $k$ is odd
- dofs:
  - evaluation at Gauß-points on the edge $\leftrightarrow$ moments against $P^{k-1}_F$ facet space
  - moments against $P^{k-3}$ space

See also [`ngstSpaceKit.CrouzeixHO`](https://johann-cm.codeberg.page/ngstspacekit/docs/ngstSpaceKit.html#CrouzeixHO)

```{code-cell} ipython3
order=3
fes = L2(mesh, order=order)
conformity_space = FacetFESpace(mesh, order=order - 1,) * L2(mesh, order=max(order - 3, 0))
u = fes.TrialFunction()
uc, vc = conformity_space.TnT()
cop_l = u * vc[1] * dx
cop_r = uc[1] * vc[1] * dx
cop_l += u * vc[0] * dx(element_vb=BND)
cop_r += uc[0] * vc[0] * dx(element_vb=BND)
embedding = TrefftzEmbedding(cop=cop_l, crhs=cop_r, ndof_trefftz=0)
crho = EmbeddedTrefftzFES(embedding)
draw_basis(crho)
```

### Argyris
- $P^5$ space
- dofs:
  - evaluation at vertices of:
    - function value
    - first order derivaties
    - second order derivatives
  - normal derivative moment on each edge, against $P^0_F$


See also [`ngstSpaceKit.Argyris`](https://johann-cm.codeberg.page/ngstspacekit/docs/ngstSpaceKit.html#Argyris)

```{code-cell} ipython3
:tags: [hide-input]

fes = L2(mesh, order=5)
normal_deriv_moment_space = NormalFacetFESpace(mesh, order=0)
conformity_space = ( H1(mesh,order=1)**6
                     * normal_deriv_moment_space)
u = fes.TrialFunction()
((u_, u_dx, u_dy, u_dxx, u_dxy, u_dyy), u_n) = conformity_space.TrialFunction()
((v_, v_dx, v_dy, v_dxx, v_dxy, v_dyy), v_n) = conformity_space.TestFunction()
dVertex = dx(element_vb=BBND)
dFace = dx(element_vb=BND)
n = specialcf.normal(2)
cop_lhs = ( u * v_ * dVertex
            + del_x(u) * v_dx * dVertex + del_y(u) * v_dy * dVertex
            + del_xx(u) * v_dxx * dVertex + del_xy(u) * v_dxy * dVertex + del_yy(u) * v_dyy * dVertex
        + grad(u) * n * v_n * n * dFace )
cop_rhs = ( u_ * v_ * dVertex
            + u_dx * v_dx * dVertex + u_dy * v_dy * dVertex
            + u_dxx * v_dxx * dVertex + u_dxy * v_dxy * dVertex + u_dyy * v_dyy * dVertex
            + u_n * n * v_n * n * dFace )
embedding = TrefftzEmbedding(cop=cop_lhs, crhs=cop_rhs, ndof_trefftz=0)
argyris = EmbeddedTrefftzFES(embedding)
draw_basis(argyris, mini=-0.1, maxi=0.1, height=.5)
```

### Crouzeix-Raviart style harmonic polynomials
- $P^4$ space
- dofs:
  - evaluation at Gauß-points on the edge $\leftrightarrow$ moments against $P^{2}_F$ facet space
  - locally enforce $\Delta u = 0$

```{code-cell} ipython3
:tags: [hide-input]

mesh = Mesh(unit_square.GenerateMesh(maxh=0.35))

base = L2(mesh, order=4, dgjumps=True)
test = L2(mesh, order=2, dgjumps=True)
conformity_space = FacetFESpace(mesh, order=2) 
u = base.TrialFunction()
q = test.TestFunction()
uc, vc = conformity_space.TnT()
cop_l = u * vc * dx(element_vb=BND)
cop_r = uc * vc * dx(element_vb=BND)

op = (-Lap(u)) * q * dx
emb = TrefftzEmbedding(op,cop=cop_l, crhs=cop_r)

print(f"full polynomial dofs: {base.ndof}")
print(f"embedded Trefftz dofs: {emb.GetEmbedding().shape[1]}")
```

```{code-cell} ipython3
:tags: [hide-input]

ut = emb.GetEmbedding().CreateRowVector()
up = GridFunction(base, multidim=len(ut))

for i, vec in enumerate(up.vecs):
    ut[:] = 0
    ut[i] = 1
    vec.data = emb.Embed(ut)

Draw(up,mesh,"basis",animate=True,interpolate_multidim=False,min=-1, max=1, deformation=True, scale=.3, euler_angles=[-70,0.4,2],)
```

```{code-cell} ipython3
:tags: [hide-input]

def laplace(fes, bndc, rhs=0, alpha=4):
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
    #a += alpha * order**2 / h * jump_u * jump_v * dx(skeleton=True)
    #a += (-mean_dudn * jump_v - mean_dvdn * jump_u) * dx(skeleton=True)
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

```{code-cell} ipython3
:tags: [hide-input]

exact = exp(x) * sin(y)

tfes = EmbeddedTrefftzFES(emb)

a_emb, f_emb = laplace(tfes, exact)
gfu_t = GridFunction(tfes)

with TaskManager():
    gfu_t.vec.data = a_emb.mat.Inverse(inverse="sparsecholesky") * f_emb.vec

gfu = GridFunction(base)
gfu.vec.data = emb.Embed(gfu_t.vec)

print(f"error: {sqrt(Integrate((gfu - exact) ** 2, mesh)):.3e}")

Draw(gfu, mesh, "projected embedded solve");
```
