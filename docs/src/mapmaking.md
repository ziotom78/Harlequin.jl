# Map-making

Harlequin includes a routine that applies the destriping algorithm to
time-ordered data, producing a maximum-likelihood map of the intensity
and polarization signal.

The destriping algorithm used in the code is described in the paper
[*Destiping CMB temperature and polarization
maps*](https://dx.doi.org/10.1051/0004-6361/200912361), Kurki-Suonio
et al., A/A 506, 1511-1539 (2009), and the source code closely follows
the terminology introduced in that paper.

The destriping algorithm is effective in removing 1/f noise originated
within the detectors of an instrument, provided that the noise is
uncorrelated among detectors. It requires the user to specify a
*baseline*, i.e., the maximum time span in which the noise can be
assumed to be uncorrelated (i.e., white).

Since the destriper is effective only when much data is available, it
is often the case that the input data to be fed to the algorithm is
larger than the available memory on a machine. In this case, the
destriper can take advantage of a distributed-memory environment and
of the MPI libraries.

## High-level functions

```@docs
destripe!
```

## Low-level functions

```@docs
```
