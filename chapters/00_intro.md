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

# Trefftz Methods in NGSolve

 ### [Trefftz Workshop](https://trefftz2026.univie.ac.at/) 7-9 September 2026 · Vienna, Austria

This book is a demo for Trefftz and embedded Trefftz methods in NGSolve/NGSTrefftz. 
The package provides implementations of
* several Trefftz spaces and quasi-Trefftz spaces
* Trefftz-DG on tent-pitched meshes for the acoustic wave equation using [ngstents](https://github.com/jayggg/ngstents)
* Embedded Trefftz method

**Follow along:** https://github.com/PaulSt/ngst26

## Overview

We focus the examples on results from the recent publications:

* *Embedded Trefftz DG method for steady Navier-Stokes flow. Part II: Nonlinear problem*  
Paul Stocker, Igor Voulis, Christoph Lehrenfeld, Philip L. Lederer  
[![arXiv](https://img.shields.io/badge/arXiv-2606.13219-b31b1b.svg)](https://arxiv.org/abs/2606.13219)
* *Embedded Trefftz DG method for steady Navier-Stokes flow. Part I: Oseen linearization*  
Paul Stocker, Igor Voulis, Christoph Lehrenfeld, Philip L. Lederer  
[![arXiv](https://img.shields.io/badge/arXiv-2606.13229-b31b1b.svg)](https://arxiv.org/abs/2606.13229)
* *Embedded Trefftz DG method for reaction-diffusion problems on anisotropic meshes*  
Sergio Gómez, Chiara Perinati, Paul Stocker, Igor Voulis  
[![arXiv](https://img.shields.io/badge/arXiv-2606.03845-b31b1b.svg)](https://arxiv.org/abs/2606.03845)
* *Embedded Trefftz DG method for the Helmholtz equation*  
Paul Stocker, Igor Voulis  
[![arXiv](https://img.shields.io/badge/arXiv-2603.13034-b31b1b.svg)](https://arxiv.org/abs/2603.13034)
* *On the Conforming Trefftz Finite Element Method and Applications*  
Johann Carl Meyer, [Master's thesis](https://zenodo.org/records/17307511)

## Contributors

* [Paul Stocker](https://github.com/PaulSt)
* [Johann Carl Meyer](https://github.com/johann-cm)
* [Christoph Lehrenfeld](https://github.com/schruste)
* [Constanze Heil](https://github.com/constanzeheil)
* [Henry von Wahl](https://github.com/hvonwah)
* [Chiara Perinati](https://github.com/ChiaraPerinati)

```{code-cell} ipython3
:tags: [hide-input]

from datetime import datetime, timedelta
from itertools import accumulate

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

commit_csv = """month,count
2017-08,6
2017-09,28
2017-10,19
2017-11,15
2017-12,13
2018-01,9
2018-02,4
2018-03,11
2018-04,14
2018-05,1
2018-06,19
2018-07,24
2018-08,16
2018-09,11
2018-10,54
2018-11,17
2018-12,22
2019-01,41
2019-02,11
2019-03,4
2019-04,42
2019-05,8
2019-06,2
2019-07,0
2019-08,2
2019-09,0
2019-10,8
2019-11,15
2019-12,5
2020-01,31
2020-02,13
2020-03,1
2020-04,3
2020-05,6
2020-06,28
2020-07,5
2020-08,8
2020-09,14
2020-10,3
2020-11,14
2020-12,6
2021-01,4
2021-02,1
2021-03,5
2021-04,0
2021-05,0
2021-06,11
2021-07,0
2021-08,34
2021-09,4
2021-10,0
2021-11,34
2021-12,23
2022-01,32
2022-02,34
2022-03,23
2022-04,8
2022-05,2
2022-06,6
2022-07,0
2022-08,14
2022-09,6
2022-10,27
2022-11,5
2022-12,0
2023-01,4
2023-02,8
2023-03,3
2023-04,10
2023-05,1
2023-06,4
2023-07,15
2023-08,4
2023-09,2
2023-10,6
2023-11,10
2023-12,10
2024-01,7
2024-02,22
2024-03,13
2024-04,6
2024-05,36
2024-06,54
2024-07,36
2024-08,7
2024-09,26
2024-10,17
2024-11,6
2024-12,0
2025-01,26
2025-02,13
2025-03,12
2025-04,37
2025-05,4
2025-06,18
2025-07,3
2025-08,3
2025-09,19
2025-10,9
2025-11,6
2025-12,0
2026-01,5
2026-02,9
2026-03,5
2026-04,7
2026-05,0
2026-06,7
"""

rows = [line.split(",") for line in commit_csv.strip().splitlines()[1:]]
months = [datetime.strptime(month, "%Y-%m") for month, _ in rows]
monthly_commits = [int(count) for _, count in rows]
cumulative_commits = list(accumulate(monthly_commits))

fig, ax = plt.subplots(figsize=(10, 3.2))
ax.bar(
    months,
    monthly_commits,
    width=25,
    color="#4e79a7",
    alpha=0.85,
    label="Monthly commits",
)
ax.set_ylabel("Monthly commits")
ax.set_title("NGSTrefftz commits over time")
ax.grid(axis="y", alpha=0.25)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(months[0] - timedelta(days=25), months[-1] + timedelta(days=25))

ax2 = ax.twinx()
ax2.plot(
    months,
    cumulative_commits,
    color="#e15759",
    linewidth=2.2,
    label="Cumulative commits",
)
ax2.set_ylabel("Cumulative commits")
ax2.set_ylim(0, max(cumulative_commits) * 1.08)

handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles + handles2, labels + labels2, loc="upper left", frameon=False)
fig.tight_layout()
```
