"""fpes — shared code for the "Fruitful PES Explorer" notebooks.

HCN <-> HNC on a rigid two-coordinate grid (theta, r) around the C-N centre of mass,
several electronic-structure fidelities (xTB, MACE, PBE, B3LYP, Skala, CCSD(T)),
a grid "oracle" so that acquisition experiments are fast and reproducible,
baseline acquisition policies, and two LLM "Judge" designs (step agent, rule agent).

Energies are stored in Hartree; barriers are reported in kcal/mol.
"""
from __future__ import annotations
import os, json, time, math, dataclasses, textwrap, warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass

HARTREE_KCAL = 627.5094740631
EV_HARTREE = 1.0 / 27.211386245988
BOHR = 0.529177210903

# ----------------------------------------------------------------------------- geometry
M_C, M_N = 12.0, 14.003074
R_CN_DEFAULT = 1.16  # Angstrom, between HCN (1.153) and HNC (1.169)


def hcn_geometry(theta_deg: float, r: float, R_CN: float = R_CN_DEFAULT):
    """Return [(symbol, (x,y,z)), ...] in Angstrom.

    C and N lie on the z axis with their centre of mass at the origin.
    theta = 0 puts H on the C side (HCN), theta = 180 on the N side (HNC).
    r is the H distance from the C-N centre of mass.
    """
    zc = -R_CN * M_N / (M_C + M_N)
    zn = R_CN * M_C / (M_C + M_N)
    t = math.radians(theta_deg)
    h = (r * math.sin(t), 0.0, -r * math.cos(t))
    return [("C", (0.0, 0.0, zc)), ("N", (0.0, 0.0, zn)), ("H", h)]


def atoms_to_pyscf(atoms):
    return "; ".join(f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in atoms)


def make_grid(n_theta: int, n_r: int, theta_range=(0.0, 180.0), r_range=(1.05, 2.30)):
    theta = np.linspace(*theta_range, n_theta)
    r = np.linspace(*r_range, n_r)
    return theta, r


# ----------------------------------------------------------------------------- hardware plan
def detect_gpu():
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out.split("\n")[0] if out else None
    except Exception:
        return None


def plan_for_hardware(gpu_name: str | None):
    """Grid density and basis per accelerator so that each notebook fits ~3 h."""
    g = (gpu_name or "").upper()
    if "A100" in g or "H100" in g:
        return dict(tier="A100", n_theta=37, n_r=25, basis="def2-tzvp", skala_every=1)
    if "L4" in g:
        return dict(tier="L4", n_theta=31, n_r=21, basis="def2-svp", skala_every=1)
    if "T4" in g:
        return dict(tier="T4", n_theta=25, n_r=17, basis="def2-svp", skala_every=1)
    return dict(tier="CPU", n_theta=13, n_r=9, basis="def2-svp", skala_every=2)


# ----------------------------------------------------------------------------- calculators
class Calc:
    """Uniform interface: calc(atoms) -> dict(E=hartree, converged=bool, niter=int, seconds=float)."""
    name = "base"

    def __call__(self, atoms):
        raise NotImplementedError


class XTB(Calc):
    name = "xtb"

    def __init__(self, method="GFN2-xTB"):
        from tblite.interface import Calculator  # noqa
        self._Calculator = Calculator
        self.method = method
        self._Z = {"H": 1, "C": 6, "N": 7, "O": 8}

    def __call__(self, atoms):
        t = time.time()
        numbers = np.array([self._Z[s] for s, _ in atoms])
        pos = np.array([c for _, c in atoms]) / BOHR
        calc = self._Calculator(self.method, numbers, pos)
        calc.set("verbosity", 0)
        try:
            res = calc.singlepoint()
            e = float(res.get("energy"))
            ok = True
        except Exception:
            e, ok = float("nan"), False
        return dict(E=e, converged=ok, niter=-1, seconds=time.time() - t)


class MACE(Calc):
    name = "mace"

    def __init__(self, model="small", device="cpu"):
        from mace.calculators import mace_off  # noqa
        from ase import Atoms  # noqa
        self._Atoms = Atoms
        self.calc = mace_off(model=model, device=device, default_dtype="float64")

    def __call__(self, atoms):
        t = time.time()
        a = self._Atoms([s for s, _ in atoms], positions=[c for _, c in atoms])
        a.calc = self.calc
        e = a.get_potential_energy() * EV_HARTREE
        return dict(E=float(e), converged=True, niter=-1, seconds=time.time() - t)


class DFT(Calc):
    """PySCF Kohn-Sham. xc in {'pbe','b3lyp',...}. gpu=True uses gpu4pyscf when importable."""

    def __init__(self, xc="pbe", basis="def2-svp", gpu=False, grids_level=3):
        from pyscf import gto, dft  # noqa
        self.gto, self.dft = gto, dft
        self.xc, self.basis, self.gpu, self.grids_level = xc, basis, gpu, grids_level
        self.name = xc
        self._dm0 = None
        if gpu:
            try:
                import gpu4pyscf  # noqa
                self._gpu_ok = True
            except Exception:
                self._gpu_ok = False
        else:
            self._gpu_ok = False

    def reset_guess(self):
        self._dm0 = None

    def __call__(self, atoms):
        t = time.time()
        mol = self.gto.M(atom=atoms_to_pyscf(atoms), basis=self.basis, unit="Angstrom", cart=False, verbose=0)
        mf = self.dft.RKS(mol)
        mf.xc = self.xc
        mf.grids.level = self.grids_level
        if self._gpu_ok:
            mf = mf.to_gpu()
        mf.conv_tol = 1e-9
        mf.max_cycle = 100
        try:
            e = mf.kernel(dm0=self._dm0)
            ok = bool(mf.converged)
            dm = mf.make_rdm1()
            self._dm0 = dm.get() if hasattr(dm, "get") else dm
            niter = int(getattr(mf, "cycles", -1) or -1)
        except Exception:
            e, ok, niter = float("nan"), False, -1
            self._dm0 = None
        return dict(E=float(e), converged=ok, niter=niter, seconds=time.time() - t)


class Skala(Calc):
    """Microsoft Skala learned XC through its PySCF bindings.

    CPU: from skala.pyscf import SkalaKS ; GPU: from skala.gpu4pyscf import SkalaKS.
    Weights download from Hugging Face on first use.
    """
    name = "skala"

    def __init__(self, basis="def2-svp", gpu=False, xc="skala-1.1", with_dftd3=False):
        from pyscf import gto  # noqa
        self.gto = gto
        if gpu:
            from skala.gpu4pyscf import SkalaKS  # noqa
        else:
            from skala.pyscf import SkalaKS  # noqa
        self.SkalaKS, self.basis, self.xc, self.with_dftd3 = SkalaKS, basis, xc, with_dftd3
        self._dm0 = None

    def reset_guess(self):
        self._dm0 = None

    def __call__(self, atoms):
        t = time.time()
        mol = self.gto.M(atom=atoms_to_pyscf(atoms), basis=self.basis, unit="Angstrom", cart=False, verbose=0)
        try:
            try:
                mf = self.SkalaKS(mol, xc=self.xc, with_dftd3=self.with_dftd3)
            except TypeError:  # older signature without with_dftd3
                mf = self.SkalaKS(mol, xc=self.xc)
            mf.conv_tol = 1e-9
            mf.max_cycle = 100
            e = mf.kernel(dm0=self._dm0)
            ok = bool(mf.converged)
            dm = mf.make_rdm1()
            self._dm0 = dm.get() if hasattr(dm, "get") else dm
            niter = int(getattr(mf, "cycles", -1) or -1)
        except Exception as ex:  # keep going; record the failure
            e, ok, niter = float("nan"), False, -1
            self._dm0 = None
            print("Skala failed:", repr(ex)[:200])
        return dict(E=float(e), converged=ok, niter=niter, seconds=time.time() - t)


class CCSDT(Calc):
    """CCSD(T) reference with PySCF (RHF-based). Closed-shell only."""
    name = "ccsd_t"

    def __init__(self, basis="def2-tzvp"):
        from pyscf import gto, scf, cc  # noqa
        self.gto, self.scf, self.cc, self.basis = gto, scf, cc, basis

    def __call__(self, atoms):
        t = time.time()
        mol = self.gto.M(atom=atoms_to_pyscf(atoms), basis=self.basis, unit="Angstrom", cart=False, verbose=0)
        try:
            mf = self.scf.RHF(mol).run(conv_tol=1e-10)
            mycc = self.cc.CCSD(mf).run()
            et = mycc.ccsd_t()
            e = mf.e_tot + mycc.e_corr + et
            ok = bool(mf.converged and mycc.converged)
        except Exception:
            e, ok = float("nan"), False
        return dict(E=float(e), converged=ok, niter=-1, seconds=time.time() - t)


# ----------------------------------------------------------------------------- surfaces with checkpoints
def compute_surface(calc: Calc, theta, r, path: str, every: int = 1, log_every: int = 10, max_minutes: float | None = None):
    """Evaluate calc on the (theta, r) grid; resume from an .npz checkpoint if present.

    `every` > 1 evaluates a coarser sub-grid (every k-th theta and r) and leaves the rest NaN.
    `max_minutes` stops gracefully (checkpoint kept) so a Colab session limit never loses work; rerun to resume.
    Returns dict with E (n_theta, n_r), converged, niter, seconds arrays.
    """
    nt, nr = len(theta), len(r)
    if os.path.exists(path):
        d = dict(np.load(path, allow_pickle=True))
        if d["E"].shape == (nt, nr):
            E, conv, nit, sec = d["E"], d["converged"], d["niter"], d["seconds"]
        else:
            E = np.full((nt, nr), np.nan); conv = np.zeros((nt, nr), bool); nit = -np.ones((nt, nr), int); sec = np.zeros((nt, nr))
    else:
        E = np.full((nt, nr), np.nan); conv = np.zeros((nt, nr), bool); nit = -np.ones((nt, nr), int); sec = np.zeros((nt, nr))
    done_mask = ~np.isnan(E)
    todo = [(i, j) for i in range(0, nt, every) for j in range(0, nr, every) if not done_mask[i, j]]
    print(f"[{calc.name}] grid {nt}x{nr}, every={every}: {len(todo)} points to compute, {done_mask.sum()} cached")
    t0 = time.time()
    last_i = None
    for k, (i, j) in enumerate(todo):
        if hasattr(calc, "reset_guess") and last_i != i:
            calc.reset_guess()  # new theta row: fresh SCF guess
        res = calc(hcn_geometry(theta[i], r[j]))
        E[i, j], conv[i, j], nit[i, j], sec[i, j] = res["E"], res["converged"], res["niter"], res["seconds"]
        last_i = i
        if max_minutes is not None and (time.time() - t0) / 60 > max_minutes:
            np.savez(path, E=E, converged=conv, niter=nit, seconds=sec, theta=theta, r=r)
            print(f"  time box of {max_minutes} min reached after {k+1} points; checkpoint saved, rerun this cell to resume")
            break
        if (k + 1) % log_every == 0 or k == len(todo) - 1:
            np.savez(path, E=E, converged=conv, niter=nit, seconds=sec, theta=theta, r=r)
            el = time.time() - t0
            print(f"  {k+1}/{len(todo)}  theta={theta[i]:.0f} r={r[j]:.2f}  E={E[i,j]:.6f}  "
                  f"conv={conv[i,j]}  {sec[i,j]:.1f}s/pt  elapsed {el/60:.1f} min  ETA {(el/(k+1))*(len(todo)-k-1)/60:.1f} min")
    np.savez(path, E=E, converged=conv, niter=nit, seconds=sec, theta=theta, r=r)
    return dict(E=E, converged=conv, niter=nit, seconds=sec, theta=theta, r=r)


def fill_nan_grid(E, theta, r):
    """Fill NaN entries (coarse Skala sub-grid or failed points) by bilinear interpolation."""
    from scipy.interpolate import griddata
    T, R = np.meshgrid(theta, r, indexing="ij")
    ok = ~np.isnan(E)
    if ok.all():
        return E.copy()
    pts = np.column_stack([T[ok] / 180.0, (R[ok] - r.min()) / (r.max() - r.min())])
    q = np.column_stack([T.ravel() / 180.0, (R.ravel() - r.min()) / (r.max() - r.min())])
    lin = griddata(pts, E[ok], q, method="linear")
    near = griddata(pts, E[ok], q, method="nearest")
    out = np.where(np.isnan(lin), near, lin).reshape(E.shape)
    return out


# ----------------------------------------------------------------------------- barrier on a grid
def basin_minima(E, theta):
    """Indices of the HCN (theta<45) and HNC (theta>135) minima."""
    hcn = np.where(theta < 45.0)[0]
    hnc = np.where(theta > 135.0)[0]
    sub = np.where(np.isnan(E), np.inf, E)
    i1 = np.unravel_index(np.argmin(sub[hcn]), sub[hcn].shape); a = (hcn[i1[0]], i1[1])
    i2 = np.unravel_index(np.argmin(sub[hnc]), sub[hnc].shape); b = (hnc[i2[0]], i2[1])
    return a, b


def bottleneck_barrier(E, theta):
    """Minimum-bottleneck path between the two basins on a 4-connected grid.

    Returns dict(barrier_kcal, saddle=(i,j), E_hcn, E_hnc, isomerization_kcal, threshold).
    """
    nt, nr = E.shape
    Ew = np.where(np.isnan(E), np.inf, E)
    a, b = basin_minima(E, theta)
    order = np.argsort(Ew, axis=None)
    parent = np.arange(nt * nr)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    active = np.zeros(nt * nr, bool)
    ia, ib = a[0] * nr + a[1], b[0] * nr + b[1]
    saddle = None
    for flat in order:
        if not np.isfinite(Ew.flat[flat]):
            break
        active[flat] = True
        i, j = divmod(int(flat), nr)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < nt and 0 <= nj < nr and active[ni * nr + nj]:
                ra, rb = find(flat), find(ni * nr + nj)
                if ra != rb:
                    parent[ra] = rb
        if find(ia) == find(ib):
            saddle = (i, j)
            break
    if saddle is None:
        return dict(barrier_kcal=float("nan"), saddle=None, E_hcn=E[a], E_hnc=E[b], isomerization_kcal=float("nan"))
    thr = E[saddle]
    return dict(barrier_kcal=float((thr - E[a]) * HARTREE_KCAL), saddle=saddle, hcn=a, hnc=b,
                E_hcn=float(E[a]), E_hnc=float(E[b]), isomerization_kcal=float((E[b] - E[a]) * HARTREE_KCAL),
                threshold=float(thr))


def corridor_mask(E, theta, margin_kcal=6.0):
    """Points within `margin` below the bottleneck threshold that connect the two basins
    (the reaction valley near the saddle). Used by path-aware acquisition rules."""
    info = bottleneck_barrier(E, theta)
    if info["saddle"] is None:
        return np.zeros_like(E, bool), info
    thr = info["threshold"]
    low = E <= thr + 1e-12
    # flood fill from the HCN minimum within the low set
    nt, nr = E.shape
    seen = np.zeros_like(low)
    stack = [info["hcn"]]
    while stack:
        i, j = stack.pop()
        if seen[i, j] or not low[i, j]:
            continue
        seen[i, j] = True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < nt and 0 <= nj < nr:
                stack.append((ni, nj))
    band = (E >= thr - margin_kcal / HARTREE_KCAL)
    return seen & band, info


# ----------------------------------------------------------------------------- oracle + surrogate
class GridOracle:
    """Expensive-method lookup on the dense reference grid. Counts calls."""

    def __init__(self, E_ref, theta, r, name="skala"):
        self.E, self.theta, self.r, self.name = E_ref, theta, r, name
        self.calls = 0
        self.log = []

    def nearest(self, th, rr):
        i = int(np.argmin(np.abs(self.theta - th))); j = int(np.argmin(np.abs(self.r - rr)))
        return i, j

    def __call__(self, i, j):
        self.calls += 1
        self.log.append((int(i), int(j), float(self.E[i, j])))
        return float(self.E[i, j])


def features(theta, r, idx):
    """Normalised (theta, r) features for GP regression."""
    th = theta[[i for i, _ in idx]] / 180.0
    rr = (r[[j for _, j in idx]] - r.min()) / (r.max() - r.min())
    return np.column_stack([th, rr])


def gp_predict(theta, r, idx, y, baseline=None, clip_kcal=150.0):
    """GP on sampled points -> (mean, std) over the full grid.

    If `baseline` (n_theta x n_r cheap surface, Hartree) is given, the GP models y - baseline
    (delta-learning) and the prediction adds baseline back.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
    X = features(theta, r, idx)
    y = np.asarray(y, float)
    if baseline is not None:
        y = y - np.array([baseline[i, j] for i, j in idx])
    elif clip_kcal is not None:  # hard-wall points carry no information about the barrier; soften them
        y = np.minimum(y, y.min() + clip_kcal / HARTREE_KCAL)
    mu, sd = y.mean(), max(y.std(), 1e-6)
    yn = (y - mu) / sd
    kern = C(1.0, (1e-2, 1e2)) * RBF(length_scale=[0.3, 0.3], length_scale_bounds=(0.03, 3.0)) + WhiteKernel(1e-6, (1e-9, 1e-2))
    gp = GaussianProcessRegressor(kern, normalize_y=False, n_restarts_optimizer=2, random_state=0)
    gp.fit(X, yn)
    T, R = np.meshgrid(theta, r, indexing="ij")
    Xg = np.column_stack([T.ravel() / 180.0, (R.ravel() - r.min()) / (r.max() - r.min())])
    m, s = gp.predict(Xg, return_std=True)
    mean = (m * sd + mu).reshape(T.shape)
    std = (s * sd).reshape(T.shape)
    if baseline is not None:
        mean = mean + baseline
    return mean, std


# ----------------------------------------------------------------------------- acquisition baselines
def initial_design(theta, r, n_init, rng):
    """Always include the two basin guesses (HCN, HNC) plus random points."""
    nt, nr = len(theta), len(r)
    idx = [(0, int(np.argmin(np.abs(r - 1.69)))), (nt - 1, int(np.argmin(np.abs(r - 1.53))))]
    while len(idx) < n_init:
        c = (int(rng.integers(nt)), int(rng.integers(nr)))
        if c not in idx:
            idx.append(c)
    return idx


def choose_next(policy, theta, r, idx, mean, std, xtb=None, rng=None):
    nt, nr = len(theta), len(r)
    sampled = np.zeros((nt, nr), bool)
    for i, j in idx:
        sampled[i, j] = True
    if policy == "random":
        cand = np.argwhere(~sampled)
        return tuple(cand[rng.integers(len(cand))])
    if policy == "uniform":
        # farthest-point in feature space (space-filling)
        T, R = np.meshgrid(theta / 180.0, (r - r.min()) / (r.max() - r.min()), indexing="ij")
        pts = np.column_stack([T.ravel(), R.ravel()])
        S = features(theta, r, idx)
        d = np.min(((pts[:, None, :] - S[None, :, :]) ** 2).sum(-1), axis=1).reshape(nt, nr)
        d[sampled] = -1
        return tuple(np.unravel_index(np.argmax(d), d.shape))
    if policy == "uncertainty":
        a = std.copy(); a[sampled] = -1
        return tuple(np.unravel_index(np.argmax(a), a.shape))
    if policy in ("path_uncertainty", "saddle_exploit"):
        mask, info = corridor_mask(mean, theta)
        a = std * (1.0 + 4.0 * mask)
        if policy == "saddle_exploit" and info["saddle"] is not None and not sampled[info["saddle"]]:
            if rng.random() < 0.5:
                return info["saddle"]
        a[sampled] = -1
        return tuple(np.unravel_index(np.argmax(a), a.shape))
    raise ValueError(policy)


def run_policy(policy, oracle: GridOracle, theta, r, budget, seed=0, n_init=4, baseline=None, ref_barrier=None):
    """Sequential acquisition with a GP surrogate; returns per-call barrier estimate and error."""
    rng = np.random.default_rng(seed)
    oracle.calls = 0; oracle.log = []
    idx = initial_design(theta, r, n_init, rng)
    y = [oracle(i, j) for i, j in idx]
    hist = []
    while True:
        mean, std = gp_predict(theta, r, idx, y, baseline=baseline)
        for i, j in idx:  # trust sampled points exactly
            mean[i, j] = y[idx.index((i, j))]
        est = bottleneck_barrier(mean, theta)["barrier_kcal"]
        hist.append(dict(calls=len(idx), barrier=est, error=(est - ref_barrier) if ref_barrier is not None else None))
        if len(idx) >= budget:
            break
        nxt = choose_next(policy, theta, r, idx, mean, std, rng=rng)
        idx.append(tuple(int(v) for v in nxt)); y.append(oracle(*nxt))
    return hist, idx, y


# ----------------------------------------------------------------------------- LLM Judge
STEP_SYSTEM = """You are the Judge in a multi-fidelity exploration of a chemical potential energy surface.
Goal (stated by the user): {usecase}
The surface is HCN <-> HNC on a rigid grid: theta in degrees (0 = H on carbon, HCN; 180 = H on nitrogen, HNC),
r in Angstrom (H distance from the C-N centre of mass). A cheap xTB surface of the whole grid is available for free.
Each expensive call returns one Skala energy at a grid point and costs budget. You have {budget} expensive calls in total.
Use the tools. Prefer calls that reduce uncertainty about the quantity the goal asks for. Explain each choice in one sentence.
When the budget is exhausted, or when you judge the estimate converged, call finish."""

STEP_TOOLS = [
    dict(name="state_summary", description="Current surrogate summary: estimated barrier, predicted saddle, most uncertain points, corridor points, calls used.",
         input_schema=dict(type="object", properties={}, additionalProperties=False), strict=True),
    dict(name="xtb_cut", description="Free: a 1D cut of the cheap xTB surface. axis='theta' scans theta at fixed r; axis='r' scans r at fixed theta.",
         input_schema=dict(type="object", properties=dict(axis=dict(type="string", enum=["theta", "r"]), value=dict(type="number")),
                           required=["axis", "value"], additionalProperties=False), strict=True),
    dict(name="skala_point", description="Spend one expensive call: Skala energy at the grid point nearest (theta, r).",
         input_schema=dict(type="object", properties=dict(theta=dict(type="number"), r=dict(type="number"), why=dict(type="string")),
                           required=["theta", "r", "why"], additionalProperties=False), strict=True),
    dict(name="finish", description="Stop exploring and report the final barrier estimate.",
         input_schema=dict(type="object", properties=dict(reason=dict(type="string")), required=["reason"], additionalProperties=False), strict=True),
]

RULE_SYSTEM = """You design acquisition rules for multi-fidelity exploration of a chemical potential energy surface.
Goal (stated by the user): {usecase}
Surface: HCN <-> HNC, rigid grid over theta (deg, 0=HCN side, 180=HNC side) and r (Angstrom, H distance from the C-N centre of mass).
Write ONE Python function

    def score(theta, r, mean, std, sampled, xtb, corridor, info):
        ...
        return S  # ndarray shape (len(theta), len(r)); the unsampled argmax is the next expensive call

Arguments: theta (1D), r (1D), mean/std (2D surrogate prediction in Hartree over the grid), sampled (2D bool),
xtb (2D cheap surface, Hartree), corridor (2D bool: points in the predicted reaction valley near the bottleneck),
info (dict with keys hcn, hnc, saddle as (i,j) tuples, barrier_kcal). numpy is available as np. 1 Hartree = 627.5 kcal/mol.
Return only the code in a single ```python block. Be specific to the goal; do not just return std."""


class _MockLLM:
    """Deterministic stand-in used when no API key is present so the notebooks run end to end."""

    def step(self, state, calls_left):
        if calls_left <= 0:
            return dict(tool="finish", input=dict(reason="budget exhausted"))
        if state["saddle"] is not None and not state["saddle_sampled"]:
            th, rr = state["saddle_theta_r"]
            return dict(tool="skala_point", input=dict(theta=th, r=rr, why="confirm predicted saddle"))
        th, rr = state["top_uncertain"][0]
        return dict(tool="skala_point", input=dict(theta=th, r=rr, why="largest uncertainty in corridor"))

    def rule(self):
        return textwrap.dedent("""
        def score(theta, r, mean, std, sampled, xtb, corridor, info):
            S = std * (1.0 + 4.0 * corridor)
            if info.get('saddle') is not None and not sampled[info['saddle']]:
                S[info['saddle']] += 10 * std.max()
            return S
        """)


class JudgeRun:
    """Shared machinery for both agent designs: surrogate state, tool execution, logging."""

    def __init__(self, oracle, theta, r, xtb_grid, budget, ref_barrier=None, baseline=None, seed=0, n_init=4):
        self.oracle, self.theta, self.r, self.xtb, self.budget = oracle, theta, r, xtb_grid, budget
        self.ref, self.baseline = ref_barrier, baseline
        rng = np.random.default_rng(seed)
        oracle.calls = 0; oracle.log = []
        self.idx, self.y, self.failed = [], [], set()
        for (i, j) in initial_design(theta, r, n_init, rng):
            self._eval(i, j)
        tries = 0
        while len(self.idx) < n_init and tries < 6 * n_init:   # replace failed initial points (live mode only)
            tries += 1
            c = (int(rng.integers(len(theta))), int(rng.integers(len(r))))
            if c not in self.idx and c not in self.failed:
                self._eval(*c)
        self.hist, self.decisions = [], []
        self.tokens = dict(input=0, output=0)
        self.seed = seed
        self.notes, self.menu_log, self.policy_log, self.summary, self.model = [], [], [], None, None
        self.refresh()

    def _eval(self, i, j):
        e = self.oracle(i, j)
        if np.isfinite(e):
            self.idx.append((i, j)); self.y.append(e)
            return True
        self.failed.add((i, j))
        return False

    def refresh(self):
        self.mean, self.std = gp_predict(self.theta, self.r, self.idx, self.y, baseline=self.baseline)
        for k, (i, j) in enumerate(self.idx):
            self.mean[i, j] = self.y[k]
        self.corridor, self.info = corridor_mask(self.mean, self.theta)
        est = self.info["barrier_kcal"]
        self.hist.append(dict(calls=len(self.idx), barrier=est, error=(est - self.ref) if self.ref is not None else None))

    def sampled_mask(self):
        m = np.zeros_like(self.mean, bool)
        for i, j in list(self.idx) + list(self.failed):
            m[i, j] = True
        return m

    def state(self, k=6):
        s = self.std.copy(); s[self.sampled_mask()] = -1
        top = [tuple(np.unravel_index(f, s.shape)) for f in np.argsort(s, axis=None)[::-1][:k]]
        sad = self.info.get("saddle")
        return dict(
            calls_used=len(self.idx), calls_left=self.budget - len(self.idx),
            barrier_kcal=round(float(self.info["barrier_kcal"]), 2) if np.isfinite(self.info["barrier_kcal"]) else None,
            isomerization_kcal=round(float(self.info["isomerization_kcal"]), 2) if np.isfinite(self.info["isomerization_kcal"]) else None,
            saddle=sad, saddle_theta_r=(float(self.theta[sad[0]]), float(self.r[sad[1]])) if sad else None,
            saddle_sampled=bool(self.sampled_mask()[sad]) if sad else False,
            saddle_std_kcal=round(float(self.std[sad] * HARTREE_KCAL), 2) if sad else None,
            top_uncertain=[(float(self.theta[i]), float(self.r[j])) for i, j in top],
            top_uncertain_std_kcal=[round(float(self.std[i, j] * HARTREE_KCAL), 2) for i, j in top],
            corridor_points=int(self.corridor.sum()),
            sampled=[(float(self.theta[i]), float(self.r[j]), round(float((self.y[k] - min(self.y)) * HARTREE_KCAL), 1)) for k, (i, j) in enumerate(self.idx)],
        )

    def xtb_cut(self, axis, value):
        if axis == "theta":
            j = int(np.argmin(np.abs(self.r - value)))
            prof = (self.xtb[:, j] - np.nanmin(self.xtb)) * HARTREE_KCAL
            return "theta(deg): E_rel(kcal/mol) at r=%.2f\n" % self.r[j] + ", ".join(f"{t:.0f}:{e:.1f}" for t, e in zip(self.theta, prof))
        i = int(np.argmin(np.abs(self.theta - value)))
        prof = (self.xtb[i, :] - np.nanmin(self.xtb)) * HARTREE_KCAL
        return "r(A): E_rel(kcal/mol) at theta=%.0f\n" % self.theta[i] + ", ".join(f"{rr:.2f}:{e:.1f}" for rr, e in zip(self.r, prof))

    def spend(self, th, rr, why=""):
        i, j = self.oracle.nearest(th, rr)
        if (i, j) in self.idx:
            return f"({self.theta[i]:.0f}, {self.r[j]:.2f}) already sampled; no budget spent. E_rel={((self.y[self.idx.index((i,j))]-min(self.y))*HARTREE_KCAL):.2f} kcal/mol"
        if (i, j) in self.failed:
            return f"({self.theta[i]:.0f}, {self.r[j]:.2f}) failed earlier (SCF); no budget spent. Pick another point."
        if len(self.idx) >= self.budget:
            return "budget exhausted"
        if not self._eval(i, j):
            return f"Skala SCF failed at (theta={self.theta[i]:.0f}, r={self.r[j]:.2f}); the point is excluded and no budget is counted. Calls left {self.budget-len(self.idx)}."
        e = self.y[-1]
        self.decisions.append(dict(theta=float(self.theta[i]), r=float(self.r[j]), why=why))
        self.refresh()
        return f"Skala E_rel at (theta={self.theta[i]:.0f}, r={self.r[j]:.2f}) = {((e-min(self.y))*HARTREE_KCAL):.2f} kcal/mol. New barrier estimate {self.info['barrier_kcal']:.2f} kcal/mol. Calls left {self.budget-len(self.idx)}."


# ----------------------------------------------------------------------------- provider-neutral chat (Claude tool calling or Gemini function calling)
class Chat:
    """Tiny adapter: one conversation with function calling.

    send(text=None, tool_results=None) -> (assistant_text, [ {id, name, input} ... ])
    tool_results is a list of (call_id, name, result_string) answering the previous calls.
    """
    provider = "mock"

    def __init__(self, system: str, tools=None, model: str | None = None):
        self.system, self.tools, self.model = system, tools or [], model
        self.tokens = dict(input=0, output=0)

    def send(self, text=None, tool_results=None):
        raise NotImplementedError


class AnthropicChat(Chat):
    provider = "anthropic"
    DEFAULT = "claude-opus-5-5"

    def __init__(self, system, tools=None, model=None):
        super().__init__(system, tools, model or self.DEFAULT)
        import anthropic
        self.client = anthropic.Anthropic()
        self.messages = []

    def send(self, text=None, tool_results=None):
        if tool_results:
            self.messages.append({"role": "user", "content": [dict(type="tool_result", tool_use_id=cid, content=out) for cid, _, out in tool_results]})
        if text:
            self.messages.append({"role": "user", "content": text})
        kw = dict(model=self.model, max_tokens=4000, system=self.system, messages=self.messages)
        if self.tools:
            kw["tools"] = self.tools
        resp = self.client.messages.create(**kw)
        self.tokens["input"] += resp.usage.input_tokens; self.tokens["output"] += resp.usage.output_tokens
        self.messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "refusal":
            return "[refused]", []
        txt = "".join(b.text for b in resp.content if b.type == "text")
        calls = [dict(id=b.id, name=b.name, input=dict(b.input)) for b in resp.content if b.type == "tool_use"]
        return txt, calls


# Gemini model ladder: the same ids as the C10H6 optimizer notebook. MODEL_FAST answers every Judge call;
# MODEL_PRO and MODEL_STRONG are used only if MODEL_FAST is unknown to the API (404) or keeps failing.
# Change them from a notebook with fpes.set_gemini_models(fast, pro, strong).
MODEL_FAST = "gemini-3.8-flash"
MODEL_PRO = "gemini-pro-latest"
MODEL_STRONG = "gemini-4-argon"
GEMINI_KEY_VARS = ("GEMINI_API_KEY", "GEMINI_API_KEY_2")   # a second key (optional) is used when the first is rate limited
GEMINI_UNAVAILABLE = set()   # model ids the API answered 404 for in this session; later chats skip them


def set_gemini_models(fast=None, pro=None, strong=None):
    """Set the Gemini ladder used by GeminiChat (blank or None keeps the current id)."""
    global MODEL_FAST, MODEL_PRO, MODEL_STRONG
    MODEL_FAST, MODEL_PRO, MODEL_STRONG = (fast or MODEL_FAST), (pro or MODEL_PRO), (strong or MODEL_STRONG)
    GeminiChat.DEFAULT = MODEL_FAST
    return [MODEL_FAST, MODEL_PRO, MODEL_STRONG]


def gemini_ladder(first=None):
    """Models to try, in order, without duplicates: the requested one first, then fast -> pro -> strong."""
    out = []
    for m in (first, MODEL_FAST, MODEL_PRO, MODEL_STRONG):
        if m and m not in out and m not in GEMINI_UNAVAILABLE:
            out.append(m)
    return out


def _gemini_error_kind(ex):
    """'model' = this model id is not available (go to the next model); 'transient' = rate limit / overload (wait, retry);
    'fatal' = bad request or bad key (raise)."""
    code = getattr(ex, "code", None) or getattr(ex, "status_code", None)
    msg = (str(getattr(ex, "status", "")) + " " + str(ex)).upper()
    if code == 404 or "NOT_FOUND" in msg or "IS NOT FOUND" in msg or "NOT SUPPORTED FOR GENERATECONTENT" in msg:
        return "model"
    if code in (429, 500, 502, 503, 504) or any(k in msg for k in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "OVERLOADED", "DEADLINE", "INTERNAL", "TIMEOUT", "TIMED OUT")):
        return "transient"
    return "fatal"


class GeminiChat(Chat):
    """Gemini function calling with a small safety net: key rotation, exponential back-off, model fallback.

    self.model is the model that answered the latest call (it changes only if the ladder had to move on);
    self.models_used counts calls per model so the cost column stays honest.
    """
    provider = "gemini"
    DEFAULT = MODEL_FAST
    MAX_ROUNDS_PER_MODEL = 3      # failed rounds (all keys) on one model before moving to the next model
    BACKOFF_BASE_S, BACKOFF_MAX_S = 5.0, 120.0

    def __init__(self, system, tools=None, model=None):
        super().__init__(system, tools, model or MODEL_FAST)
        from google import genai
        from google.genai import types
        self.types = types
        keys = [os.environ.get(v) for v in GEMINI_KEY_VARS] + [os.environ.get("GOOGLE_API_KEY")]
        keys = [k for i, k in enumerate(keys) if k and k not in keys[:i]]
        self.clients = [genai.Client(api_key=k) for k in keys] or [genai.Client()]
        self.client = self.clients[0]
        self.ladder = gemini_ladder(self.model)
        if not self.ladder:
            raise RuntimeError(f"Gemini: every model id was rejected earlier in this session ({sorted(GEMINI_UNAVAILABLE)}); fix the ids with fpes.set_gemini_models(...)")
        self.model = self.ladder[0]
        self.models_used, self.waited_s, self.retries = {}, 0.0, 0
        self.tokens_by_model = {}
        decls = []
        for t in self.tools:
            params = {k: v for k, v in t["input_schema"].items() if k != "additionalProperties"}
            decls.append(types.FunctionDeclaration(name=t["name"], description=t["description"], parameters=params))
        self.config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=decls)] if decls else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) if decls else None,
        )
        self.contents = []
        self._n = 0

    def _generate(self):
        """One generate_content call with key rotation, back-off and model fallback. Raises only when every model failed."""
        import random
        last = None
        for model in list(self.ladder):
            for rnd in range(self.MAX_ROUNDS_PER_MODEL):
                for client in self.clients:
                    try:
                        resp = client.models.generate_content(model=model, contents=self.contents, config=self.config)
                        if model != self.model:
                            print(f"[gemini] answered by {model} (fallback from {self.model})")
                        self.model = model
                        self.models_used[model] = self.models_used.get(model, 0) + 1
                        return resp
                    except Exception as ex:   # noqa: BLE001  (the SDK raises several error classes)
                        last, kind = ex, _gemini_error_kind(ex)
                        if kind == "fatal":
                            raise
                        if kind == "model":
                            GEMINI_UNAVAILABLE.add(model)
                            print(f"[gemini] model {model} not available ({str(ex)[:80]}); trying the next model")
                            break
                        self.retries += 1
                else:   # every key hit a transient error: wait, then another round on the same model
                    wait = min(self.BACKOFF_MAX_S, self.BACKOFF_BASE_S * 2 ** rnd) * (0.8 + 0.4 * random.random())
                    print(f"[gemini] {model}: rate limited / busy ({str(last)[:60]}); waiting {wait:.0f} s")
                    time.sleep(wait); self.waited_s += wait
                    continue
                break   # model not available: leave the rounds loop, go to the next model
        raise RuntimeError(f"Gemini: no model in {self.ladder} answered; last error: {last!r}")

    def send(self, text=None, tool_results=None):
        T = self.types
        if tool_results:
            self.contents.append(T.Content(role="user", parts=[T.Part.from_function_response(name=name, response={"result": out}) for _, name, out in tool_results]))
        if text:
            self.contents.append(T.Content(role="user", parts=[T.Part.from_text(text=text)]))
        resp = self._generate()
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            ti, to = int(um.prompt_token_count or 0), int(um.candidates_token_count or 0) + int(getattr(um, "thoughts_token_count", 0) or 0)
            self.tokens["input"] += ti; self.tokens["output"] += to
            tm = self.tokens_by_model.setdefault(self.model, dict(input=0, output=0)); tm["input"] += ti; tm["output"] += to
        cand = resp.candidates[0] if resp.candidates else None
        if cand is None or cand.content is None:
            return "", []
        self.contents.append(cand.content)      # kept whole: Gemini 3 function calling needs its thought signatures back
        txt, calls = "", []
        for part in cand.content.parts or []:
            if getattr(part, "function_call", None):
                self._n += 1
                calls.append(dict(id=f"g{self._n}", name=part.function_call.name, input=dict(part.function_call.args or {})))
            elif getattr(part, "text", None) and not getattr(part, "thought", False):
                txt += part.text
        return txt, calls


def chat_factory(provider: str = "auto", model: str | None = None):
    """Return a callable (system, tools) -> Chat, or None for mock mode.

    provider: 'auto' picks Gemini if GEMINI_API_KEY is set, else Claude if ANTHROPIC_API_KEY is set, else mock.
    Colab secrets named GEMINI_API_KEY / GEMINI_API_KEY_2 / ANTHROPIC_API_KEY are read automatically.
    model: blank = MODEL_FAST for Gemini (with the fast -> pro -> strong fallback), claude-opus-5-5 for Claude.
    """
    for var in (*GEMINI_KEY_VARS, "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            try:
                from google.colab import userdata  # noqa
                v = userdata.get(var)
                if v:
                    os.environ[var] = v
            except Exception:
                pass
    if provider == "auto":
        provider = "gemini" if os.environ.get("GEMINI_API_KEY") else ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock")
    if provider == "mock":
        print("Judge in MOCK mode (scripted policy). Add GEMINI_API_KEY or ANTHROPIC_API_KEY as a Colab secret to use a real model.")
        return None
    if provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY not set")
        n_keys = sum(bool(os.environ.get(v)) for v in GEMINI_KEY_VARS)
        print(f"Judge on Gemini: {gemini_ladder(model)} (first answers; the others are fallbacks) | API keys: {n_keys}")
        return lambda system, tools=None: GeminiChat(system, tools, model or None)
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return lambda system, tools=None: AnthropicChat(system, tools, model or None)
    raise ValueError(provider)


# $ per million tokens (input, output); used only for the cost column.
# Models not listed here show cost NaN (unknown), never a fake $0. Add a price with fpes.PRICE["model-id"] = (in, out).
PRICE = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
         "gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-pro": (1.25, 10.0)}
_PRICE_WARNED = set()


def cost_usd(model, tokens):
    if not model or str(model).startswith(("frozen", "mock")):
        return 0.0
    if model not in PRICE:
        if model not in _PRICE_WARNED:
            _PRICE_WARNED.add(model)
            print(f"[cost] no price for {model}: cost shows NaN. Set fpes.PRICE['{model}'] = (usd_per_M_in, usd_per_M_out) to fill it.")
        return float("nan")
    pi, po = PRICE[model]
    return (tokens["input"] * pi + tokens["output"] * po) / 1e6


def run_step_agent(judge: JudgeRun, usecase: str, chat_factory=None, max_turns=80, verbose=True, model=None, client=None):
    """Design A: the LLM chooses every expensive point itself, through function calls.

    chat_factory: from fpes.chat_factory(); None = scripted mock. (`client`/`model` kept for backward compatibility.)
    """
    if chat_factory is None and client is not None:
        chat_factory = lambda system, tools=None: AnthropicChat(system, tools, model)
    mock = chat_factory is None
    system = STEP_SYSTEM.format(usecase=usecase, budget=judge.budget)
    chat = None if mock else chat_factory(system, STEP_TOOLS)
    judge.model = "mock" if mock else chat.model
    pending_text, pending_results = "Begin. Call state_summary first.", None
    for turn in range(max_turns):
        if mock:
            st = judge.state()
            act = _MockLLM().step(st, st["calls_left"])
            calls = [dict(id=f"mock{turn}", name=act["tool"], input=act["input"])]
        else:
            txt, calls = chat.send(text=pending_text, tool_results=pending_results)
            judge.tokens = chat.tokens; judge.model = chat.model
            pending_text, pending_results = None, None
            if txt == "[refused]":
                print("model refused; stopping"); break
            if not calls:
                pending_text = "Continue: call a tool (skala_point, xtb_cut, state_summary) or finish."
                continue
        results, done = [], False
        for c in calls:
            name, inp = c["name"], c["input"]
            if name == "state_summary":
                out = json.dumps(judge.state())
            elif name == "xtb_cut":
                out = judge.xtb_cut(inp["axis"], float(inp["value"]))
            elif name == "skala_point":
                out = judge.spend(float(inp["theta"]), float(inp["r"]), str(inp.get("why", "")))
                if verbose:
                    print(f"  call {len(judge.idx):2d}: ({float(inp['theta']):.0f}, {float(inp['r']):.2f}) {str(inp.get('why',''))[:70]} -> {out[:60]}")
            elif name == "finish":
                out = "finished"; done = True
            else:
                out = f"unknown tool {name}"
            results.append((c["id"], name, out))
        pending_results = results
        if done or len(judge.idx) >= judge.budget:
            break
    return judge


def _compile_rule(code: str):
    code = code.strip()
    if "```" in code:
        code = code.split("```python")[-1].split("```")[0] if "```python" in code else code.split("```")[1]
    ns = {"np": np, "math": math}
    exec(compile(code, "<rule>", "exec"), ns)
    return ns["score"], code


def run_rule_agent(judge: JudgeRun, usecase: str, chat_factory=None, review_every=10, max_rewrites=3, verbose=True, model=None, client=None):
    """Design B: the LLM writes an acquisition rule once; code executes it; the LLM may rewrite after each review."""
    if chat_factory is None and client is not None:
        chat_factory = lambda system, tools=None: AnthropicChat(system, tools, model)
    mock = chat_factory is None
    chat = None if mock else chat_factory(RULE_SYSTEM.format(usecase=usecase), None)
    judge.model = "mock" if mock else chat.model
    rules = []

    def ask(prompt):
        if mock:
            return _MockLLM().rule()
        txt, _ = chat.send(text=prompt)
        judge.tokens = chat.tokens; judge.model = chat.model
        return txt

    score_fn, code = _compile_rule(ask("Write the initial acquisition rule.")); rules.append(code)
    rewrites = 0
    while len(judge.idx) < judge.budget:
        try:
            S = np.asarray(score_fn(judge.theta, judge.r, judge.mean, judge.std, judge.sampled_mask(), judge.xtb, judge.corridor, judge.info), float)
            S = np.where(np.isfinite(S), S, -np.inf)
        except Exception as ex:
            if verbose: print("rule failed, falling back to uncertainty:", repr(ex)[:100])
            S = judge.std.copy()
        S[judge.sampled_mask()] = -np.inf
        i, j = np.unravel_index(int(np.argmax(S)), S.shape)
        judge.spend(float(judge.theta[i]), float(judge.r[j]), why=f"rule v{len(rules)}")
        if verbose:
            print(f"  call {len(judge.idx):2d}: ({judge.theta[i]:.0f}, {judge.r[j]:.2f}) barrier est {judge.info['barrier_kcal']:.2f}")
        if len(judge.idx) % review_every == 0 and rewrites < max_rewrites and len(judge.idx) < judge.budget:
            review = (f"Review after {len(judge.idx)} calls. State: {json.dumps(judge.state(k=4))}. "
                      "Barrier estimates so far (kcal/mol): " + ", ".join(f"{h['barrier']:.1f}" for h in judge.hist[-review_every:] if np.isfinite(h['barrier'])) +
                      ". Keep the rule or return an improved one in a single ```python block.")
            try:
                score_fn, code = _compile_rule(ask(review)); rules.append(code); rewrites += 1
            except Exception as ex:
                if verbose: print("rewrite unusable, keeping previous rule:", repr(ex)[:100])
    judge.rules = rules
    return judge


# ----------------------------------------------------------------------------- Designs M (menu judge) and K (knob tuner)
# Design M: code builds a short, stratified menu of candidate grid points each turn; the LLM picks one by id.
# Design K: the LLM sets the weights of a fixed acquisition formula once per round; code runs the round.
# Both have a "frozen policy" twin (no LLM) that uses the same menus / the same formula, which is the ablation
# that measures the agent's contribution. Both also serve as the mock Judge when no API key is present.

_SHADES = " .:-=+*#%@"
KNOB_NAMES = ("w_unc", "w_path", "w_gap", "w_saddle", "w_spread", "w_wall")


def ascii_map(Z, theta, r, sampled, mark=None, lo=0.0, hi=80.0):
    """Small character map for the LLM: rows = r (top = largest), columns = theta. 'o' sampled, 'S' marked cell."""
    lines = []
    for j in range(len(r) - 1, -1, -1):
        row = ""
        for i in range(len(theta)):
            if mark is not None and (i, j) == tuple(mark):
                row += "S"
            elif sampled[i, j]:
                row += "o"
            else:
                v = (Z[i, j] - lo) / (hi - lo + 1e-12)
                row += _SHADES[min(9, max(0, int(v * 9.999)))] if np.isfinite(v) else "?"
        lines.append(f"r={r[j]:.2f} |{row}|")
    lines.append(" " * 8 + "θ=0" + " " * max(1, len(theta) - 7) + "θ=180")
    return "\n".join(lines)


def _norm01(a):
    a = np.where(np.isfinite(a), a, 0.0)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-12)


def judge_features(J):
    """Per-cell features: predicted E_rel and sigma (kcal/mol), |xTB - surrogate| gap (kcal/mol), grid distance to nearest sample."""
    sig = J.std * HARTREE_KCAL
    mu = (J.mean - J.mean.min()) * HARTREE_KCAL
    gap = np.abs((J.mean - J.mean.min()) - (J.xtb - np.nanmin(J.xtb))) * HARTREE_KCAL
    nt, nr = J.mean.shape
    T, R = np.meshgrid(np.arange(nt), np.arange(nr), indexing="ij")
    pts = np.column_stack([T.ravel(), R.ravel()]).astype(float)
    S = np.array(J.idx, float)
    dist = np.sqrt(((pts[:, None, :] - S[None, :, :]) ** 2).sum(-1)).min(1).reshape(nt, nr)
    return mu, sig, gap, dist


def build_menu(J, rng=None, box=None, want=None, k_max=8):
    """Stratified candidate menu: one unsampled cell per criterion, duplicates removed.

    box = (theta_min, theta_max, r_min, r_max) restricts the search (probe).
    want in {uncertainty, disagreement, low_energy, spread} ranks the whole box by one criterion instead (probe).
    """
    rng = rng if rng is not None else np.random.default_rng(getattr(J, "seed", 0))
    mu, sig, gap, dist = judge_features(J)
    allowed = ~J.sampled_mask()
    if box is not None:
        tmin, tmax, rmin, rmax = [float(x) for x in box]
        allowed &= (J.theta[:, None] >= tmin) & (J.theta[:, None] <= tmax) & (J.r[None, :] >= rmin) & (J.r[None, :] <= rmax)
    cands = []

    def add(tag, score, k=1, positive=False):
        s = np.where(allowed, score, -np.inf)
        for flat in np.argsort(s, axis=None)[::-1][:k]:
            i, j = (int(x) for x in np.unravel_index(int(flat), s.shape))
            if not np.isfinite(s[i, j]) or (positive and s[i, j] <= 0):
                break
            if all((i, j) != c[0] for c in cands):
                cands.append(((i, j), tag))

    if want is not None:
        crit = {"uncertainty": sig, "disagreement": gap, "low_energy": -mu, "spread": dist}.get(want, sig)
        add(str(want), crit, k=k_max)
    else:
        sad = J.info.get("saddle")
        if sad is not None and allowed[sad]:
            cands.append((tuple(int(x) for x in sad), "saddle guess"))
        add("max-σ on path", sig * J.corridor, positive=True)
        add("max-σ anywhere", sig)
        add("xTB≠GP on path", gap * J.corridor, positive=True)
        add("xTB≠GP anywhere", gap)
        add("far from samples", dist)
        basin_added = False
        for key in ("hcn", "hnc"):
            b = J.info.get(key)
            if b is not None and allowed[b] and all(tuple(b) != c[0] for c in cands):
                cands.append((tuple(int(x) for x in b), f"{key.upper()} basin min, unsampled")); basin_added = True; break
        if not basin_added:
            add("low E, unsampled", -mu)
        free = np.argwhere(allowed)
        if len(free):
            i, j = (int(x) for x in free[rng.integers(len(free))])
            if all((i, j) != c[0] for c in cands):
                cands.append(((i, j), "random"))
    rows = []
    for n, ((i, j), tag) in enumerate(cands[:k_max], 1):
        rows.append(dict(id=n, i=i, j=j, theta=float(J.theta[i]), r=float(J.r[j]), tag=tag,
                         E_pred=float(mu[i, j]), sigma=float(sig[i, j]), gap=float(gap[i, j]),
                         on_path=bool(J.corridor[i, j]), dist=float(dist[i, j])))
    return rows


def menu_text(rows):
    out = ["MENU (pick one by id)", " id    θ     r  why it is here              E_pred    σ   gap path dist"]
    for m in rows:
        out.append(f" [{m['id']}] {m['theta']:>3.0f}  {m['r']:.2f}  {m['tag']:<26} {m['E_pred']:>7.1f} {m['sigma']:>4.1f} {m['gap']:>5.1f}  {'Y' if m['on_path'] else 'n'}  {m['dist']:>4.1f}")
    return "\n".join(out)


def state_block(J):
    """What the LLM reads every turn: budget, estimate and trend, saddle and basins, two ASCII maps, its own notes."""
    mu, sig, gap, dist = judge_features(J)
    sampled = J.sampled_mask(); sad = J.info.get("saddle"); hcn, hnc = J.info.get("hcn"), J.info.get("hnc")
    est = J.info["barrier_kcal"]
    traj = ", ".join(f"{h['barrier']:.1f}" for h in J.hist[-6:] if np.isfinite(h["barrier"]))
    out = [f"BUDGET   used {len(J.idx)} / {J.budget}  (left {J.budget - len(J.idx)})",
           f"ESTIMATE barrier {est:.1f} kcal/mol" + (f", σ at saddle {sig[sad]:.1f}" if sad is not None else ""),
           f"         last estimates: {traj}"]
    if sad is not None:
        out.append(f"SADDLE   θ={J.theta[sad[0]]:.0f}°, r={J.r[sad[1]]:.2f} Å  ({'sampled' if sampled[sad] else 'NOT yet sampled at Skala level'})")
    if hcn is not None and hnc is not None:
        out.append(f"BASINS   HCN θ={J.theta[hcn[0]]:.0f} r={J.r[hcn[1]]:.2f} ({'sampled' if sampled[hcn] else 'unsampled'})")
        out.append(f"         HNC θ={J.theta[hnc[0]]:.0f} r={J.r[hnc[1]]:.2f} ({'sampled' if sampled[hnc] else 'unsampled'}); ΔE(HNC−HCN) {J.info['isomerization_kcal']:.1f}")
    if J.decisions:
        d = J.decisions[-1]
        out.append(f"LAST     ({d['theta']:.0f}, {d['r']:.2f}) because: {str(d['why'])[:120]}")
    out += ["", "MAP predicted E_rel, kcal/mol  (' '=0 … '@'≥80)", "    o = sampled, S = saddle guess",
            ascii_map(mu, J.theta, J.r, sampled, sad, 0, 80),
            "", "MAP uncertainty σ, kcal/mol  (' '=0 … '@'≥10)",
            ascii_map(sig, J.theta, J.r, sampled, sad, 0, 10),
            "", "NOTES (yours): " + ("; ".join(J.notes[-6:]) if J.notes else "none yet")]
    return "\n".join(out)


def pick_from_menu(J, menu, cid, why):
    """Spend one expensive call on menu item `cid`; log which kind of candidate was taken."""
    try:
        cid = int(cid)
    except Exception:
        return f"id must be an integer 1..{len(menu)}"
    m = next((x for x in menu if x["id"] == cid), None)
    if m is None:
        return f"no candidate with id {cid}; ids are 1..{len(menu)}"
    before = J.info["barrier_kcal"]
    out = J.spend(m["theta"], m["r"], str(why))
    J.menu_log.append(dict(call=len(J.idx), id=cid, tag=m["tag"], theta=m["theta"], r=m["r"], why=str(why)[:200],
                           offered=[x["tag"] for x in menu]))
    sad = J.info.get("saddle")
    extra = f" Barrier was {before:.1f}."
    if sad is not None:
        extra += f" Saddle guess now θ={J.theta[sad[0]]:.0f}, r={J.r[sad[1]]:.2f}; σ there {J.std[sad] * HARTREE_KCAL:.1f}."
    return out + extra


MENU_ORDER = """
DEFAULT DECISION ORDER (the GOAL overrides it; take the first that applies):
 1. Saddle guess unsampled and sigma there above 0.5: pick it.
 2. An on-path candidate has sigma above 1 or gap above 3: pick the one that most threatens the
    estimate (largest sigma + gap on the path).
 3. A basin minimum is unsampled: pick it; the barrier is measured from it.
 4. Estimate moved less than 0.5 over the last 5 picks and nothing above applies: finish.
 5. Otherwise the most uncertain candidate with E_pred below 80.
"""

MENU_SYSTEM = """You are the Judge of a multi-fidelity exploration of a chemical potential energy surface.

GOAL (from the user, verbatim): {usecase}
METRIC: the HCN -> HNC bottleneck barrier (the highest energy on the lowest connecting path
across the grid), within 1 kcal/mol of the converged answer, using as few expensive calls as
possible. Both count: the error and the number of calls.

THE SURFACE: HCN <-> HNC on a rigid grid. theta (degrees): 0 = H on the carbon side (HCN),
180 = H on the nitrogen side (HNC). r (Angstrom): distance of H from the C-N centre of mass.
Expect two basins (theta near 0-15 and near 165-180), one saddle around theta 70-100 at small r,
and steep walls (E_rel above 80 kcal/mol) at small r inside the basins and at large r everywhere.
The cheap xTB surface is roughly right about shape and often wrong about heights near the saddle.

FREE: the state block, the maps, the menu, probe, xtb_cut, note.  COSTS: pick, one Skala energy.
You have {budget} picks.

EACH TURN you get a state block (budget, estimate and its trend, saddle guess, two ASCII maps:
predicted E_rel and uncertainty sigma; 'o' = sampled, 'S' = saddle guess) and a MENU of
candidates with: E_pred, sigma, gap = |E_rel from xTB - E_rel from the surrogate|,
path = Y if the point lies in the reaction valley near the bottleneck, dist = grid steps to the
nearest sampled point.
{order}
RULES: exactly one function call per turn. Never pick E_pred above 80 unless the GOAL is about
walls. Never pick a point marked already sampled. Start 'why' with what the last result changed,
then the reason for this pick; 30 words at most. Call note() when you form or drop a hypothesis;
notes are shown back to you. A good estimate with budget left beats a spent budget."""

MENU_TOOLS = [
    dict(name="pick", description="Spend one expensive call on a menu candidate. The only function that costs budget.",
         input_schema=dict(type="object", properties=dict(id=dict(type="integer", description="menu id"), why=dict(type="string", description="what the last result changed, then the reason; 30 words max")),
                           required=["id", "why"], additionalProperties=False), strict=True),
    dict(name="probe", description="Free: rebuild the menu inside a box of the grid, ranked by one criterion.",
         input_schema=dict(type="object", properties=dict(theta_min=dict(type="number"), theta_max=dict(type="number"), r_min=dict(type="number"), r_max=dict(type="number"),
                                                          want=dict(type="string", enum=["uncertainty", "disagreement", "low_energy", "spread"])),
                           required=["theta_min", "theta_max", "r_min", "r_max", "want"], additionalProperties=False), strict=True),
    STEP_TOOLS[1],  # xtb_cut
    dict(name="note", description="Free: store a short hypothesis or observation; it is shown back to you in every later state block.",
         input_schema=dict(type="object", properties=dict(text=dict(type="string")), required=["text"], additionalProperties=False), strict=True),
    dict(name="finish", description="Stop exploring and report the final barrier estimate.",
         input_schema=dict(type="object", properties=dict(barrier_kcal=dict(type="number"), confidence=dict(type="string", enum=["high", "medium", "low"]), summary=dict(type="string")),
                           required=["barrier_kcal", "confidence", "summary"], additionalProperties=False), strict=True),
]

KNOB_SYSTEM = """You set the sampling policy of a multi-fidelity exploration of a chemical potential energy surface.

GOAL (from the user, verbatim): {usecase}
METRIC: the HCN -> HNC bottleneck barrier (the highest energy on the lowest connecting path
across the grid), within 1 kcal/mol of the converged answer, using as few expensive calls as
possible. Both count: the error and the number of calls.

THE SURFACE: HCN <-> HNC on a rigid grid. theta (degrees): 0 = H on the carbon side (HCN),
180 = H on the nitrogen side (HNC). r (Angstrom): distance of H from the C-N centre of mass.
Expect two basins (theta near 0-15 and near 165-180), one saddle around theta 70-100 at small r,
and steep walls (E_rel above 80 kcal/mol) at small r inside the basins and at large r everywhere.
The cheap xTB surface is roughly right about shape and often wrong about heights near the saddle.

FREE: the review block and xtb_cut.  COSTS: set_policy runs round_size Skala calls at once.
You have {budget} calls in total. Code picks each point by maximising
  score = w_unc*sigma + w_path*(sigma on the path) + w_gap*|xTB - GP| + w_saddle*[saddle guess]
        + w_spread*(distance to sampled points) - w_wall*[E_pred above 80]
with every term scaled 0..1 over the grid. You choose the six weights (0 to 2) and round_size (4-10).

DEFAULT SCHEDULE (the GOAL overrides it): early rounds locate (w_saddle, w_path high, w_spread
moderate, large rounds); middle rounds hunt disagreement (w_gap up); late rounds confirm (w_path
only, small rounds). Finish when the barrier trajectory is flat within 0.5 over two rounds.

FORMAT: one function call per turn. In 'why', name the number in the review that made you change
each weight you changed. Keeping the same weights is allowed; say why."""

KNOB_TOOLS = [
    dict(name="set_policy", description="Set the acquisition weights and run one round of expensive calls. Returns a review of the round.",
         input_schema=dict(type="object", properties=dict(w_unc=dict(type="number"), w_path=dict(type="number"), w_gap=dict(type="number"), w_saddle=dict(type="number"),
                                                          w_spread=dict(type="number"), w_wall=dict(type="number"), round_size=dict(type="integer"), why=dict(type="string")),
                           required=list(KNOB_NAMES) + ["round_size", "why"], additionalProperties=False), strict=True),
    STEP_TOOLS[1],  # xtb_cut
    MENU_TOOLS[-1],  # finish
]


def acquisition(J, w):
    """Weighted acquisition over the grid (design K); sampled cells are -inf."""
    mu, sig, gap, dist = judge_features(J)
    sad = np.zeros_like(sig)
    if J.info.get("saddle") is not None:
        sad[J.info["saddle"]] = 1.0
    A = (w.get("w_unc", 0.0) * _norm01(sig) + w.get("w_path", 0.0) * _norm01(sig * J.corridor) + w.get("w_gap", 0.0) * _norm01(gap)
         + w.get("w_saddle", 0.0) * sad + w.get("w_spread", 0.0) * _norm01(dist) - w.get("w_wall", 0.0) * (mu > 80))
    A[J.sampled_mask()] = -np.inf
    return A


def _sigma_path_max(J):
    sig = J.std * HARTREE_KCAL
    return float((sig * J.corridor).max()) if J.corridor.any() else float(sig.max())


def run_round(J, w, n, why=""):
    """Run one round of design K with weights w; returns the review block for the LLM."""
    w = {k: float(min(2.0, max(0.0, float(w.get(k, 0.0))))) for k in KNOB_NAMES}
    n = int(min(12, max(1, int(n))))
    before = _sigma_path_max(J)
    rows = []
    label = " ".join(f"{k}={v:g}" for k, v in w.items())
    for _ in range(n):
        if len(J.idx) >= J.budget:
            break
        A = acquisition(J, w)
        if not np.isfinite(A).any():
            break
        i, j = np.unravel_index(int(np.argmax(A)), A.shape)
        J.spend(float(J.theta[i]), float(J.r[j]), why=f"round {len(J.policy_log) + 1}: {label}")
        rows.append((float(J.theta[i]), float(J.r[j]), float((J.y[-1] - min(J.y)) * HARTREE_KCAL), float(J.info["barrier_kcal"])))
    after = _sigma_path_max(J)
    J.policy_log.append(dict(round=len(J.policy_log) + 1, weights=w, round_size=n, why=str(why)[:300], calls=len(J.idx),
                             barrier=float(J.info["barrier_kcal"]), sigma_path_before=before, sigma_path_after=after))
    return review_block(J, w, rows, before, after)


def review_block(J, w, rows, before, after):
    sampled = J.sampled_mask(); sad = J.info.get("saddle"); sig = J.std * HARTREE_KCAL
    out = ["ROUND DONE with weights", "  " + " ".join(f"{k}={v:g}" for k, v in w.items()),
           f"BUDGET used {len(J.idx)} / {J.budget}",
           "POINTS THIS ROUND  (θ, r) -> E_rel ; barrier estimate after"]
    out += [f"  ({th:>3.0f}, {rr:.2f}) -> {e:>6.1f} ; {b:.1f}" for th, rr, e, b in rows]
    out.append("BARRIER trajectory (all rounds):")
    out.append("  " + ", ".join(f"{h['barrier']:.1f}" for h in J.hist[-14:] if np.isfinite(h["barrier"])))
    out.append(f"σ on path: max {before:.1f} -> {after:.1f} kcal/mol")
    if sad is not None:
        out.append(f"saddle guess θ={J.theta[sad[0]]:.0f} r={J.r[sad[1]]:.2f} ({'sampled' if sampled[sad] else 'unsampled'}), σ there {sig[sad]:.1f}")
    out.append("MAP σ (o=sampled, S=saddle guess)")
    out.append(ascii_map(sig, J.theta, J.r, sampled, sad, 0, 10))
    out.append(f"Calls left {J.budget - len(J.idx)}. Reply with set_policy(...) for the next round, or finish(...).")
    return "\n".join(out)


def _flat(J, k=5, tol=0.5):
    b = [h["barrier"] for h in J.hist[-(k + 1):] if np.isfinite(h["barrier"])]
    return len(b) >= k + 1 and (max(b) - min(b)) < tol


def frozen_menu_choice(J, menu, rule="order"):
    """Chooser without an LLM. rule='order' follows MENU_ORDER literally (mock Judge and frozen twin);
    rule='max_sigma' always takes the most uncertain candidate below the wall."""
    def act(m, why):
        return dict(name="pick", input=dict(id=m["id"], why=why))
    below = [m for m in menu if m["E_pred"] < 80] or menu
    if rule == "max_sigma":
        return act(max(below, key=lambda m: m["sigma"]), "frozen: most uncertain candidate")
    sad = next((m for m in menu if m["tag"] == "saddle guess"), None)
    if sad is not None and sad["sigma"] > 0.5:
        return act(sad, "rule 1: saddle guess unsampled")
    path = [m for m in menu if m["on_path"] and (m["sigma"] > 1 or m["gap"] > 3)]
    if path:
        return act(max(path, key=lambda m: m["sigma"] + m["gap"]), "rule 2: on-path candidate threatens the estimate")
    basin = next((m for m in menu if "basin" in m["tag"]), None)
    if basin is not None:
        return act(basin, "rule 3: basin minimum unsampled")
    if _flat(J):
        return dict(name="finish", input=dict(barrier_kcal=float(J.info["barrier_kcal"]), confidence="medium", summary="rule 4: estimate flat over the last 5 picks"))
    return act(max(below, key=lambda m: m["sigma"]), "rule 5: most uncertain candidate below the wall")


def frozen_knob_schedule(J, round_no):
    """Weights without an LLM: the default schedule of KNOB_SYSTEM (mock Judge and frozen twin)."""
    if round_no == 1:
        return dict(w_unc=0.2, w_path=1.0, w_gap=0.3, w_saddle=1.0, w_spread=0.3, w_wall=1.0, round_size=8, why="frozen schedule: locate")
    if round_no == 2:
        return dict(w_unc=0.1, w_path=1.5, w_gap=0.6, w_saddle=0.5, w_spread=0.0, w_wall=1.0, round_size=6, why="frozen schedule: hunt disagreement")
    return dict(w_unc=0.0, w_path=1.0, w_gap=0.2, w_saddle=0.0, w_spread=0.0, w_wall=1.0, round_size=6, why="frozen schedule: confirm")


def run_menu_agent(J, usecase, chat_factory=None, scaffold=True, frozen=None, max_free=3, verbose=True):
    """Design M: the LLM picks one tagged candidate per turn from a code-built menu.

    chat_factory None, or frozen in {'order', 'max_sigma'}: no LLM, the frozen chooser decides (mock mode / frozen twin).
    scaffold False removes the default decision order from the prompt (ablation).
    Each pick is a fresh conversation; memory is the notes and the state block, so cost stays flat.
    """
    rng = np.random.default_rng(getattr(J, "seed", 0))
    system = MENU_SYSTEM.format(usecase=usecase, budget=J.budget, order=MENU_ORDER if scaffold else "\n")
    use_llm = chat_factory is not None and frozen is None
    J.model = None if use_llm else f"frozen-{frozen or 'order'}"
    done = False
    while len(J.idx) < J.budget and not done:
        menu = build_menu(J, rng)
        if not menu:
            break
        chat = None
        if use_llm:
            chat = chat_factory(system, MENU_TOOLS); J.model = chat.model
            txt, calls = chat.send(text=state_block(J) + "\n\n" + menu_text(menu))
        else:
            a = frozen_menu_choice(J, menu, frozen or "order")
            calls = [dict(id="frozen", name=a["name"], input=a["input"])]
        free = 0
        for _guard in range(12):
            if not calls:
                if chat is None:
                    break
                txt, calls = chat.send(text="Call exactly one function: pick, probe, xtb_cut, note or finish.")
                continue
            c = calls[0]; name, inp = c["name"], c["input"]
            if name == "pick":
                out = pick_from_menu(J, menu, inp.get("id", 0), inp.get("why", ""))
                if "no candidate" in out or "must be an integer" in out:
                    if chat is None:
                        break
                    txt, calls = chat.send(tool_results=[(c["id"], name, out)]); continue
                if verbose:
                    print(f"  call {len(J.idx):2d}: {out[:70]} | {str(inp.get('why', ''))[:60]}")
                break
            if name == "finish":
                J.summary = dict(inp); done = True
                if verbose:
                    print(f"  finish after {len(J.idx)} calls: {str(inp.get('summary', ''))[:80]}")
                break
            if name == "probe":
                try:
                    menu = build_menu(J, rng, box=(inp["theta_min"], inp["theta_max"], inp["r_min"], inp["r_max"]), want=inp.get("want"))
                    out = menu_text(menu) if menu else "no unsampled points in that box"
                except Exception as ex:
                    out = f"probe failed: {ex!r}"
            elif name == "xtb_cut":
                out = J.xtb_cut(inp.get("axis", "theta"), float(inp.get("value", 0)))
            elif name == "note":
                J.notes.append(str(inp.get("text", ""))[:200]); out = "noted"
            else:
                out = f"unknown function {name}"
            free += 1
            if free >= max_free:
                out += "\nFree calls for this turn are used up: pick or finish now."
            if chat is None:
                break
            txt, calls = chat.send(tool_results=[(c["id"], name, out)])
        if chat is not None:
            J.tokens["input"] += chat.tokens["input"]; J.tokens["output"] += chat.tokens["output"]; J.model = chat.model
    return J


def run_knob_agent(J, usecase, chat_factory=None, fixed_weights=None, max_rounds=8, verbose=True):
    """Design K: the LLM sets six acquisition weights per round; code runs the round and returns a review.

    chat_factory None: the frozen default schedule decides (mock mode / frozen twin).
    fixed_weights: the same weights every round (a notebook-2 baseline expressed as a policy).
    """
    use_llm = chat_factory is not None and fixed_weights is None
    system = KNOB_SYSTEM.format(usecase=usecase, budget=J.budget)
    chat = chat_factory(system, KNOB_TOOLS) if use_llm else None
    J.model = chat.model if use_llm else ("frozen-fixed" if fixed_weights else "frozen-schedule")
    pending_text = "Begin. The landscape before any policy:\n\n" + state_block(J) + "\n\nCall set_policy for round 1."
    pending_results = None
    rounds = 0
    for _turn in range(max_rounds * 3):
        if len(J.idx) >= J.budget or rounds >= max_rounds:
            break
        if use_llm:
            txt, calls = chat.send(text=pending_text, tool_results=pending_results)
            pending_text, pending_results = None, None
            J.tokens = chat.tokens; J.model = chat.model
            if not calls:
                pending_text = "Call set_policy or finish."; continue
            c = calls[0]; name, inp = c["name"], c["input"]
        else:
            inp = dict(fixed_weights, round_size=int(fixed_weights.get("round_size", 6)), why="fixed weights") if fixed_weights else frozen_knob_schedule(J, rounds + 1)
            name, c = "set_policy", dict(id="frozen")
        if name == "finish":
            J.summary = dict(inp)
            if verbose:
                print(f"  finish after {len(J.idx)} calls: {str(inp.get('summary', ''))[:80]}")
            break
        if name == "xtb_cut":
            pending_results = [(c["id"], name, J.xtb_cut(inp.get("axis", "theta"), float(inp.get("value", 0))))]; continue
        if name != "set_policy":
            pending_results = [(c["id"], name, "unknown function; call set_policy or finish")]; continue
        review = run_round(J, inp, inp.get("round_size", 6), inp.get("why", ""))
        rounds += 1
        if verbose:
            p = J.policy_log[-1]
            print(f"  round {p['round']}: {' '.join(f'{k}={v:g}' for k, v in p['weights'].items())} n={p['round_size']} -> barrier {p['barrier']:.2f}, σ_path {p['sigma_path_after']:.1f} | {p['why'][:60]}")
        pending_results = [(c["id"], "set_policy", review)]
        if not use_llm and len(J.policy_log) >= 3 and abs(J.policy_log[-1]["barrier"] - J.policy_log[-2]["barrier"]) < 0.5 and abs(J.policy_log[-2]["barrier"] - J.policy_log[-3]["barrier"]) < 0.5:
            J.summary = dict(barrier_kcal=float(J.info["barrier_kcal"]), confidence="medium", summary="frozen schedule: flat over two rounds")
            break
    return J


def anthropic_client_or_none():
    """Backward-compatible helper: returns an Anthropic client or None (mock)."""
    f = chat_factory("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock")
    if f is None:
        return None
    import anthropic
    return anthropic.Anthropic()


# ----------------------------------------------------------------------------- plotting
def plot_surface(ax, E, theta, r, idx=None, title="", levels=None, cmap="viridis", vmax_kcal=120):
    Er = (E - np.nanmin(E)) * HARTREE_KCAL
    Er = np.clip(Er, 0, vmax_kcal)
    T, R = np.meshgrid(theta, r, indexing="ij")
    cs = ax.contourf(T, R, Er, levels=levels or np.linspace(0, vmax_kcal, 25), cmap=cmap)
    ax.contour(T, R, Er, levels=[10, 20, 30, 40, 50, 60], colors="w", linewidths=0.5, alpha=0.6)
    if idx is not None:
        ax.scatter([theta[i] for i, _ in idx], [r[j] for _, j in idx], s=18, c="#F0A04B", edgecolor="k", linewidth=0.4, zorder=5)
    ax.set_xlabel("theta (deg)   0 = HCN, 180 = HNC"); ax.set_ylabel("r (Angstrom)"); ax.set_title(title)
    return cs


def summarize_curves(runs, budget):
    """runs: list of hist lists (one per seed). Returns calls, median |error|, q25, q75 arrays."""
    calls = np.arange(min(len(h) for h in runs))
    err = np.array([[abs(h[k]["error"]) if h[k]["error"] is not None and np.isfinite(h[k]["error"]) else np.nan for k in calls] for h in runs])
    return np.array([h[k]["calls"] for k in calls for h in runs[:1]]), np.nanmedian(err, 0), np.nanpercentile(err, 25, 0), np.nanpercentile(err, 75, 0)


# ----------------------------------------------------------------------------- minimax path (for NB4)
def bottleneck_path(E, theta):
    """Minimax (lowest-maximum) path between the two basin minima on the grid, as a list of (i, j)."""
    import heapq
    nt, nr = E.shape
    Ew = np.where(np.isnan(E), np.inf, E)
    a, b = basin_minima(E, theta)
    best = np.full((nt, nr), np.inf); prev = {}
    best[a] = Ew[a]
    pq = [(Ew[a], a)]
    while pq:
        c, (i, j) = heapq.heappop(pq)
        if c > best[i, j]:
            continue
        if (i, j) == b:
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < nt and 0 <= nj < nr:
                nc = max(c, Ew[ni, nj])
                if nc < best[ni, nj]:
                    best[ni, nj] = nc; prev[(ni, nj)] = (i, j); heapq.heappush(pq, (nc, (ni, nj)))
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


class LiveOracle(GridOracle):
    """Same interface as GridOracle but calls a real calculator at the grid geometry (live GPU mode).

    Every point is computed at most once and cached (optionally on disk), so several Judge runs can share it:
    `fresh_calls` / `fresh_seconds` count only real calculator calls. A point whose SCF fails or does not
    converge is retried once from a fresh guess; if it still fails it is marked failed and returns NaN, which
    JudgeRun excludes from the surrogate and from later menus.
    """

    def __init__(self, calc: Calc, theta, r, name="skala-live", cache_path=None, accept_unconverged=False, retries=1):
        shape = (len(theta), len(r))
        super().__init__(np.full(shape, np.nan), theta, r, name)
        self.calc, self.cache_path, self.accept_unconverged, self.retries = calc, cache_path, accept_unconverged, retries
        self.conv = np.zeros(shape, bool); self.sec = np.zeros(shape); self.failed = np.zeros(shape, bool)
        self.fresh_calls, self.fresh_seconds = 0, 0.0
        if cache_path and os.path.exists(cache_path):
            d = dict(np.load(cache_path, allow_pickle=True))
            if d["E"].shape == shape:
                self.E, self.conv, self.sec, self.failed = d["E"], d["conv"], d["sec"], d["failed"]
                print(f"[{name}] cache: {int((~np.isnan(self.E)).sum())} points loaded, {int(self.failed.sum())} marked failed")

    def save(self):
        if self.cache_path:
            np.savez(self.cache_path, E=self.E, conv=self.conv, sec=self.sec, failed=self.failed, theta=self.theta, r=self.r)

    def compute(self, i, j):
        """Real calculator call for cell (i, j) unless cached; returns the energy (NaN if failed)."""
        if np.isnan(self.E[i, j]) and not self.failed[i, j]:
            ok = False
            for _attempt in range(1 + self.retries):
                if hasattr(self.calc, "reset_guess"):
                    self.calc.reset_guess()
                res = self.calc(hcn_geometry(self.theta[i], self.r[j]))
                self.fresh_calls += 1; self.fresh_seconds += res["seconds"]
                if np.isfinite(res["E"]) and (res["converged"] or self.accept_unconverged):
                    self.E[i, j], self.conv[i, j], self.sec[i, j] = res["E"], res["converged"], res["seconds"]
                    ok = True
                    break
            if not ok:
                self.failed[i, j] = True
                print(f"  [{self.name}] SCF failed at theta={self.theta[i]:.0f} r={self.r[j]:.2f}; point excluded")
            self.save()
        return float(self.E[i, j])

    def __call__(self, i, j):
        self.compute(i, j)
        return super().__call__(i, j)


# ----------------------------------------------------------------------------- other small systems (NB4)
def nh3_geometry(umbrella_deg: float, r_NH: float = 1.012):
    """NH3 with the three H at angle `umbrella_deg` below the plane (0 = planar transition state, ~22 = equilibrium)."""
    t = math.radians(umbrella_deg)
    rho, z = r_NH * math.cos(t), -r_NH * math.sin(t)
    atoms = [("N", (0.0, 0.0, 0.0))]
    for k in range(3):
        p = 2 * math.pi * k / 3
        atoms.append(("H", (rho * math.cos(p), rho * math.sin(p), z)))
    return atoms


def ch4_geometry(stretch: float, r_CH: float = 1.087):
    """Methane with one C-H bond stretched to `stretch` Angstrom; the other three fixed at tetrahedral positions."""
    v = 1 / math.sqrt(3)
    dirs = [(v, v, v), (-v, -v, v), (-v, v, -v), (v, -v, -v)]
    atoms = [("C", (0.0, 0.0, 0.0))]
    for k, d in enumerate(dirs):
        L = stretch if k == 0 else r_CH
        atoms.append(("H", (d[0] * L, d[1] * L, d[2] * L)))
    return atoms


def scan_1d(geoms, calcs: dict, path: str):
    """Evaluate each calculator on a list of geometries; checkpoint to JSON; return {name: [E,...]}."""
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name, calc in calcs.items():
        rec = out.get(name, {"E": [], "converged": [], "seconds": []})
        if hasattr(calc, "reset_guess"):
            calc.reset_guess()
        for k in range(len(rec["E"]), len(geoms)):
            res = calc(geoms[k])
            rec["E"].append(res["E"]); rec["converged"].append(res["converged"]); rec["seconds"].append(res["seconds"])
            out[name] = rec
            json.dump(out, open(path, "w"))
            print(f"  [{name}] point {k+1}/{len(geoms)}  E={res['E']:.6f} conv={res['converged']} {res['seconds']:.1f}s")
    return out
