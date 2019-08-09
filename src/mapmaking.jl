# -*- encoding: utf-8 -*-

export PolarizedMap, NobsMatrixElement, Observation, DestripingData, CleanedTOD
export update_nobs!, update_nobs_matrix!, observation, compute_nobs_matrix!
export update_binned_map!, reset_maps!, compute_z_and_subgroup!, compute_z_and_group!
export compute_residuals!, array_dot, calc_stopping_factor, apply_offset_to_baselines!
export calculate_cleaned_map!, destripe!

import LinearAlgebra: I, Symmetric, cond, diagm, dot
import RunLengthArrays: RunLengthArray, runs, values
import Healpix
import Logging: @debug, @info, @warn
import Logging: ConsoleLogger, global_logger, Info, Debug
import Statistics: mean, std
import Printf: @sprintf

################################################################################

# This implementation of the destriping algorithm is based on the
# paper «Destriping CMB temperature and polarization maps» by
# Kurki-Suonio et al. 2009, A&A 506, 1511–1539 (2009),
# https://dx.doi.org/10.1051/0004-6361/200912361
#
# It is important to have that paper at hand while reading this code,
# as many functions and variable defined here use the same letters and
# symbols of that paper. We refer to it in code comments and
# docstrings as "KurkiSuonio2009".

################################################################################

@doc raw"""
    mutable struct PolarizedMap{T, O <: Healpix.Order}

A polarized I/Q/U map. It contains three Healpix maps with the same NSIDE:

- `i`
- `q`
- `u`

You can create an instance of this type using the function
[`PolarizedMap{T,O}`](@ref).

"""
mutable struct PolarizedMap{T, O <: Healpix.Order}
    i::Healpix.Map{T,O}
    q::Healpix.Map{T,O}
    u::Healpix.Map{T,O}

    PolarizedMap{T, O}(
        i::Healpix.Map{T, O},
        q::Healpix.Map{T, O},
        u::Healpix.Map{T, O},
    ) where {T, O <: Healpix.Order} = new(i, q, u)
end

function PolarizedMap{T, O}(
    i::AbstractVector{T},
    q::AbstractVector{T},
    u::AbstractVector{T},
) where {T, O <: Healpix.Order}

    @assert length(i) == length(q)
    @assert length(i) == length(u)

    PolarizedMap{T, O}(
        Healpix.Map{T, O}(i),
        Healpix.Map{T, O}(q),
        Healpix.Map{T, O}(u),
    )
end

################################################################################
# Matrix containing the number of observations and the condition
# number per pixel

@doc raw"""
    mutable struct NobsMatrixElement

This structure encodes the condition matrix for a pixel in the map. It
is essentially matrix M_p in Eq. (10) of KurkiSuonio2009, but with
steroids, as it implements the following fields:

- `m`: the 3×3 matrix in Eq. (10)
- `invm`: the 3×3 inverse of `m`
- `invcond`: the inverse of the condition number *rcond* for `m`; this
  field is useful to check whether the samples in a TOD and the attack
  angles are sufficient to constrain a unique solution for I, Q, and U
  for that pixel.
- `neglected`: a Boolean flag telling whether the pixel was skipped or
  not during map-making (this usually happens if `invcond` is too
  small)

A typical use case is an array of `NobsMatrixElement` objects in an
array that get updated via [`update_nobs_matrix!`](@ref), like in the
following example:

```julia
NPIX = 10   # Number of pixels in the map
nobs_matrix = [NobsMatrixElement() for i in 1:NPIX]

# The variables psi_angle, sigma_squared, pixidx, and flagged
# must have been defined somewhere else; they contain the TOD
update_nobs_matrix!(nobs_matrix, psi_angle, sigma_squared, pixidx, flagged)
```

"""
mutable struct NobsMatrixElement
    # We do not define "m" as symmetric, as we want to update its
    # fields one by one; however, we only consider the upper part
    m::Array{Float64, 2}
    invm::Symmetric{Float64, Array{Float64, 2}}
    invcond::Float64

    # If "true", do not consider this pixel in the solution, as it was
    # not covered enough by the scanning strategy
    neglected::Bool

    NobsMatrixElement() = new(
        Symmetric(zeros(Float64, 3, 3)),
        Symmetric(zeros(Float64, 3, 3)),
        0.0,
        true,
    )
end

function Base.show(io::IO, nobsmatr::NobsMatrixElement)
    print(io, "NobsMatrixElement(invcond => $(nobsmatr.invcond), neglected => $(nobsmatr.neglected))")
end

function Base.show(io::IO, ::MIME"text/plain", nobsmatr::NobsMatrixElement)
    println(
        io,
        @sprintf(
            """NobsMatrixElement:
Inverse condition number.......... %e
Neglected......................... %s
Matrix:
    %.5f   %.5f   %.5f
    %.5f   %.5f   %.5f
    %.5f   %.5f   %.5f
Inverse matrix:
    %.5f   %.5f   %.5f
    %.5f   %.5f   %.5f
    %.5f   %.5f   %.5f
    """,
            nobsmatr.invcond,
            nobsmatr.neglected,
            nobsmatr.m[1, 1], nobsmatr.m[1, 2], nobsmatr.m[1, 3],
            nobsmatr.m[2, 1], nobsmatr.m[2, 2], nobsmatr.m[2, 3],
            nobsmatr.m[3, 1], nobsmatr.m[3, 2], nobsmatr.m[3, 3],
            nobsmatr.invm[1, 1], nobsmatr.invm[1, 2], nobsmatr.invm[1, 3],
            nobsmatr.invm[2, 1], nobsmatr.invm[2, 2], nobsmatr.invm[2, 3],
            nobsmatr.invm[3, 1], nobsmatr.invm[3, 2], nobsmatr.invm[3, 3],
        )
    )
end

@doc raw"""
    update_nobs!(nobs::NobsMatrixElement; threshold = 1e-7)

This function makes sure that all the elements in `nobs` are
synchronized. It should be called whenever the field `nobs.m` (matrix
M_p in Eq. 10 of KurkiSuonio2009) has changed.

"""
function update_nobs!(nobs::NobsMatrixElement; threshold = 1e-7)
    c = cond(nobs.m)

    if isfinite(c)
        nobs.invm = inv(Symmetric(nobs.m))
        nobs.invcond = 1 / c
        nobs.neglected = nobs.invcond < threshold
    else
        nobs.invcond = 0.0
        nobs.neglected = true
    end
end

@doc raw"""
    update_nobs_matrix!(nobs_matrix::Vector{NobsMatrixElement}, psi_angle, sigma_squared, pixidx, flagged)

Apply Eq. (10) of KurkiSuonio2009 iteratively on the samples of a TOD
to update matrices ``M_p`` in `nobs_matrix`. The meaning of the
parameters is the following:

- `nobs_matrix` is the structure that gets updated by this function
- `psi_angle` is an array of `N` elements, containing the polarization
  angles (in radians)
- `sigma_squared` is an array of `N` elements, each containing the
  value of σ^2 for the samples in the TOD
- `pixidx` is an array of `N` elements, containing the pixel index
  observed by the TOD
- `flagged` is a Boolean array of `N` elements; `true` means that the
  sample in the TOD should not be used to produce the map. This can be
  used to produce jackknives and to neglect moving objects in the TOD

"""
function update_nobs_matrix!(
    nobs_matrix::Vector{NobsMatrixElement},
    psi_angle,
    sigma_squared,
    pixidx,
    flagged,
)

    @assert length(psi_angle) == length(sigma_squared)
    @assert length(psi_angle) == length(pixidx)
    @assert length(psi_angle) == length(flagged)

    for (idx, curpix, curpsi, curflagged) in zip(
        1:length(pixidx),
        pixidx,
        psi_angle,
        flagged,
    )
        curflagged && continue

        constant = 1 / sigma_squared[idx]

        sin_term, cos_term = sincos(2 * curpsi)
        nobs_matrix[curpix].m[1, 1] += constant
        nobs_matrix[curpix].m[1, 2] += cos_term * constant
        nobs_matrix[curpix].m[1, 3] += sin_term * constant

        nobs_matrix[curpix].m[2, 2] += cos_term^2 * constant
        nobs_matrix[curpix].m[2, 3] += cos_term * sin_term * constant

        nobs_matrix[curpix].m[3, 3] += sin_term^2 * constant
    end
end

include("mapmaking_observations.jl")

@doc raw"""
    compute_nobs_matrix!(nobs_matrix::Vector{NobsMatrixElement}, observations::Vector{Observation{T}})

Initialize all the elements in `nobs_matrix` so that they are the
matrices M_p in Eq. (10) of KurkiSuonio2009. The TOD is taken from the
list of observations in the parameter `observations`.

"""
function compute_nobs_matrix!(
    nobs_matrix::Vector{NobsMatrixElement},
    observations::Vector{Observation{T}},
) where {T}

    for submatrix in nobs_matrix
        submatrix.m .= 0.0
    end

    for idx in 1:length(observations)
        update_nobs_matrix!(
            nobs_matrix,
            observations[idx].psi_angle,
            observations[idx].sigma_squared,
            observations[idx].pixidx,
            observations[idx].flagged,
        )
    end

    for idx in 1:length(nobs_matrix)
        update_nobs!(nobs_matrix[idx])
    end
end

################################################################################

function update_binned_map!(
    vec,
    skymap::PolarizedMap{T, O},
    hitmap::Healpix.Map{T, O},
    obs::Observation{T};
    comm = nothing,
    unseen = missing,
) where {T <: Real, O <: Healpix.Order}

    # We use "zip" to iterate over those arrays that might not be
    # plain Julia Arrays
    for (idx, vecsamp, sigmasamp, mapidx) in zip(1:length(vec),
                                                 vec,
                                                 obs.sigma_squared,
                                                 obs.pixidx)

        sin_term, cos_term = sincos(2 * obs.psi_angle[idx])

        skymap.i[mapidx] += vecsamp / sigmasamp
        skymap.q[mapidx] += vecsamp * cos_term / sigmasamp
        skymap.u[mapidx] += vecsamp * sin_term / sigmasamp
        hitmap[mapidx] += 1 / sigmasamp
    end

    if comm != nothing
        skymap.i .= MPI.allreduce(skymap.i, MPI.SUM, comm)
        skymap.q .= MPI.allreduce(skymap.q, MPI.SUM, comm)
        skymap.u .= MPI.allreduce(skymap.u, MPI.SUM, comm)
        hitmap[:] = MPI.allreduce(hitmap, MPI.SUM, comm)
    end
end

function update_binned_map!(
    skymap::PolarizedMap{T, O},
    hitmap::Healpix.Map{T, O},
    obs::Observation{T};
    comm = nothing,
    unseen = missing,
) where {T <: Real, O <: Healpix.Order}

    update_binned_map!(
        obs.tod,
        skymap,
        hitmap,
        obs,
        comm = comm,
        unseen = unseen,
    )
end

function update_binned_map!(
    vec_list,
    skymap::PolarizedMap{T, O},
    hitmap::Healpix.Map{T, O},
    obs_list::Vector{Observation{T}};
    comm = nothing,
    unseen = missing,
) where {T <: Real, O <: Healpix.Order}

    @assert length(vec_list) == length(obs_list)

    for (cur_vec, cur_obs) in zip(vec_list, obs_list)
        update_binned_map!(
            cur_vec,
            skymap,
            hitmap,
            cur_obs,
            comm = comm,
            unseen = unseen,
        )
    end
end

function update_binned_map!(
    skymap::PolarizedMap{T, O},
    hitmap::Healpix.Map{T, O},
    obs_list::Vector{Observation{T}};
    comm = nothing,
    unseen = missing,
) where {T <: Real, O <: Healpix.Order}

    for cur_obs in obs_list
        update_binned_map!(
            skymap,
            hitmap,
            cur_obs,
            comm = comm,
            unseen = unseen,
        )
    end
end


@doc raw"""
    update_binned_map!(vec, skymap::PolarizedMap{T, O}, hitmap::Healpix.Map{T, O}, obs::Observation{T}; comm=nothing, unseen=NaN) where {T <: Real, O <: Healpix.Order}
    update_binned_map!(skymap::PolarizedMap{T, O}, hitmap::Healpix.Map{T, O}, obs::Observation{T}; comm=nothing, unseen=NaN) where {T <: Real, O <: Healpix.Order}
    update_binned_map!(skymap::PolarizedMap{T, O}, hitmap::Healpix.Map{T, O}, obs_list::Vector{Observation{T}}; comm=nothing, unseen=NaN) where {T <: Real, O <: Healpix.Order}

Apply Eqs. (18), (19), and (20) to compute the values of I, Q, and U
from a TOD, and save the result in `skymap` and `hitmap`.

The three versions differ for the source of the TOD calibrated data:

- if the `vec` parameter is present (first form), it will be used
  instead of `obs.tod` (second form)
- instead of `obs` (a single [`Observation`](@ref) object), you can
  pass an array `obs_list` (third form). All the elements in this
  array will be used to update `skymap` and `hitmap`.

"""
update_binned_map!


################################################################################

include("mapmaking_destripingdata.jl")

include("mapmaking_cleanedtod.jl")

################################################################################

function compute_z_and_subgroup!(
    dest_baselines,
    vec,
    baseline_lengths,
    destriping_data::DestripingData{T, O},
    obs::Observation{T};
    comm = nothing,
    unseen = missing,
) where {T, O <: Healpix.Order}

    # WARNING: unlike compute_z_and_group!, this does *not* update
    # `destriping_data`!

    length(baseline_lengths) > 0 || return
    @assert sum(baseline_lengths) == length(obs.tod)
    @assert length(baseline_lengths) == length(dest_baselines)

    iqu = Array{Float64}(undef, 3)

    @inbounds @simd for idx in eachindex(dest_baselines)
        dest_baselines[idx] = 0.0
    end

    baseline_idx = 1
    samples_in_baseline = 0
    for (idx, sigmasamp, mapsamp) in zip(eachindex(obs.pixidx),
                                         obs.sigma_squared,
                                         obs.pixidx)
        iqu[1] = destriping_data.skymap.i[mapsamp]
        iqu[2] = destriping_data.skymap.q[mapsamp]
        iqu[3] = destriping_data.skymap.u[mapsamp]

        iqu .= destriping_data.nobs_matrix[mapsamp].invm * iqu
        sin_term, cos_term = sincos(2 * obs.psi_angle[idx])
        signal = (iqu[1] + iqu[2] * cos_term + iqu[3] * sin_term)
        dest_baselines[baseline_idx] += (vec[idx] - signal) / sigmasamp

        samples_in_baseline += 1
        if samples_in_baseline == baseline_lengths[baseline_idx]
            baseline_idx += 1
            samples_in_baseline = 0
        end
    end
end

function compute_z_and_subgroup!(
    dest_baselines,
    baseline_lengths,
    destriping_data::DestripingData{T, O},
    obs::Observation{T};
    comm = nothing,
    unseen = missing,
) where {T, O <: Healpix.Order}

    # WARNING: unlike compute_z_and_group!, this does *not* update
    # `destriping_data`!

    compute_z_and_group!(
        dest_baselines,
        obs.tod,
        baseline_lengths,
        destriping_data,
        obs,
        comm = comm,
        unseen = unseen,
    )
end

@doc raw"""
    compute_z_and_subgroup!(dest_baselines, vec, baseline_lengths, destriping_data::DestripingData{T, O}, obs::Observation{T}; comm=nothing, unseen=NaN) where {T, O <: Healpix.Order}
    compute_z_and_subgroup!(dest_baselines, baseline_lengths, destriping_data::DestripingData{T, O}, obs::Observation{T}; comm=nothing, unseen=NaN) where {T, O <: Healpix.Order}

Compute the application of matrix ``A`` in Eq. (25) in
KurkiSuonio2009. The result is a set of baselines, and it is saved in
`dest_baselines` (an array of elements whose type is `T`). The two
forms differ only in the presence of the `vec` parameter: if it is not
present, the value of `obs.tod` will be used.

This function only operates on single observations. If you want to
apply it to an array of observations, use
[`compute_z_and_group!`](@ref).

"""
compute_z_and_subgroup!

function compute_z_and_group!(
    dest_baselines,
    vectors,
    baseline_lengths,
    destriping_data::DestripingData{T, O},
    obs_list::Vector{Observation{T}};
    comm = nothing,
    unseen = missing,
) where {T, O <: Healpix.Order}

    @assert length(obs_list) == length(vectors)
    @assert length(obs_list) == length(baseline_lengths)
    @assert length(obs_list) == length(dest_baselines)

    for (idx, cur_obs) in enumerate(obs_list)
        update_binned_map!(
            vectors[idx],
            destriping_data.skymap,
            destriping_data.hitmap,
            cur_obs,
            comm = comm,
            unseen = unseen,
        )
    end

    for (idx, cur_obs, cur_baseline_length) in zip(1:length(obs_list), obs_list, baseline_lengths)
        @assert length(cur_baseline_length) == length(dest_baselines[idx])

        compute_z_and_subgroup!(
            dest_baselines[idx],
            vectors[idx],
            cur_baseline_length,
            destriping_data,
            cur_obs,
            comm = comm,
            unseen = unseen,
        )
    end
end

function compute_z_and_group!(
    dest_baselines,
    baseline_lengths,
    destriping_data::DestripingData{T, O},
    obs_list::Vector{Observation{T}};
    comm = nothing,
    unseen = missing,
) where {T, O <: Healpix.Order}

    @assert length(obs_list) == length(baseline_lengths)
    @assert length(obs_list) == length(dest_baselines)

    for cur_obs in obs_list
        update_binned_map!(
            cur_obs.tod,
            destriping_data.skymap,
            destriping_data.hitmap,
            cur_obs,
            comm = comm,
            unseen = unseen,
        )
    end

    for (idx, cur_obs, cur_baseline_length) in zip(1:length(obs_list), obs_list, baseline_lengths)
        @assert length(cur_baseline_length) == length(dest_baselines[idx])

        compute_z_and_subgroup!(
            dest_baselines[idx],
            cur_obs.tod,
            cur_baseline_length,
            destriping_data,
            cur_obs,
            comm = comm,
            unseen = unseen,
        )
    end
end


@doc raw"""
    compute_z_and_group!(dest_baselines, vectors, baseline_lengths, destriping_data::DestripingData{T, O}, obs_list::Vector{Observation{T}}; comm=nothing, unseen=NaN) where {T, O <: Healpix.Order}
    compute_z_and_group!(dest_baselines, baseline_lengths, destriping_data::DestripingData{T, O}, obs_list::Vector{Observation{T}}; comm=nothing, unseen=NaN) where {T, O <: Healpix.Order}

Compute the application of matrix ``A`` in Eq. (25) in
KurkiSuonio2009. The result is a set of baselines, and it is saved in
`dest_baselines` (an array of elements whose type is `T`). The two
forms differ only in the presence of the `vec` parameter: if it is not
present, the value of `obs.tod` will be used.

"""
compute_z_and_group!

################################################################################

@doc raw"""
    compute_residuals!(residuals, baseline_lengths, destriping_data, obs_list; comm=nothing, unseen=missing)

This function is used to compute the value of ``A x - b`` at the
beginning of the Conjugated-Gradient algorithm implemented in
[`destripe!`](@ref).

"""
function compute_residuals!(
    residuals,
    baseline_lengths,
    destriping_data::DestripingData{T, O},
    obs_list::Vector{Observation{T}};
    comm = nothing,
    unseen = missing,
) where {T, O <: Healpix.Order}

    @assert length(residuals) == length(obs_list)
    @assert length(residuals) == length(baseline_lengths)

    reset_maps!(destriping_data)

    compute_z_and_group!(
        residuals,
        baseline_lengths,
        destriping_data,
        obs_list,
    )

    left_side = [Array{T}(undef, length(residuals[idx])) for idx in 1:length(residuals)]

    reset_maps!(destriping_data)

    compute_z_and_group!(
        left_side,
        destriping_data.baselines,
        baseline_lengths,
        destriping_data,
        obs_list,
    )
end

################################################################################

function array_dot(
    x::Array{RunLengthArray{N, T}},
    y;
    comm = nothing,
) where {N <: Integer, T <: Real}

    @assert length(x) == length(y)

    result = 0.0
    for (curx, cury) in zip(x, y)
        result += dot(values(curx), cury)
    end

    result
end

function array_dot(x, y; comm = nothing)
    @assert length(x) == length(y)

    result = 0.0
    for (curx, cury) in zip(x, y)
        result += dot(curx, cury)
    end

    result
end

@doc raw"""
    array_dot(x, y; comm = nothing)

Compute the dot product between `x` and `y`, assuming that both are
arrays of arrays.

"""
array_dot

@doc raw"""
    calc_stopping_factor(r; comm=nothing)

Calculate the stopping factor required by the Conjugated Gradient
method to determine if the iteration must be stopped or not. The
parameter `r` is the vector of the residuals.

"""
calc_stopping_factor(r; comm = nothing) = maximum(abs.(Iterators.flatten(r)))

################################################################################

@doc raw"""
    apply_offset_to_baselines!(baselines)

Add a constant offset to all the baselines such that their sum is zero.

"""
function apply_offset_to_baselines!(
    baselines::Array{RunLengthArray{Int, T}, 1},
) where {T <: Real}

    baseline_sum = 0.0
    baseline_num = 0
    for cur_baseline in baselines
        baseline_sum += sum(values(cur_baseline))
        baseline_num += length(values(cur_baseline))
    end
    baseline_sum /= baseline_num

    for cur_baseline in baselines
        cur_baseline.values .-= baseline_sum
    end
end

@doc raw"""
    calculate_cleaned_map!(obs_list, destriping_data; comm=nothing, unseen=missing)

Provided a set of baselines in `baselines`, this function cleans the
TOD in `obs_list` and computes the I/Q/U maps, which are saved in
`destriping_data`.

"""
function calculate_cleaned_map!(
    obs_list::Vector{Observation{T}},
    destriping_data::DestripingData{T, O};
    comm = nothing,
    unseen = missing,
) where {T <: Real, O <: Healpix.Order}

    cleaned_tod = CleanedTOD{Float64}([], [])
    reset_maps!(destriping_data)

    for (cur_obs, cur_baseline) in zip(obs_list, destriping_data.baselines)
        cleaned_tod.tod = cur_obs.tod
        cleaned_tod.baselines = cur_baseline

        update_binned_map!(
            cleaned_tod,
            destriping_data.skymap,
            destriping_data.hitmap,
            cur_obs,
            comm = comm,
            unseen = unseen,
        )
    end

    # Now solve for the I, Q, U parameters
    iqu = Array{Float64}(undef, 3)
    for curpix in eachindex(destriping_data.nobs_matrix)
        iqu[1] = destriping_data.skymap.i[curpix]
        iqu[2] = destriping_data.skymap.q[curpix]
        iqu[3] = destriping_data.skymap.u[curpix]

        iqu .= destriping_data.nobs_matrix[curpix].invm * iqu

        destriping_data.skymap.i[curpix] = iqu[1]
        destriping_data.skymap.q[curpix] = iqu[2]
        destriping_data.skymap.u[curpix] = iqu[3]
    end
end

include("mapmaking_preconditioners.jl")
include("mapmaking_covmatrix.jl")

@doc raw"""
    destripe!(obs_list, destriping_data::DestripingData{T,O}; comm, unseen, callback) where {T <: Real, O <: Healpix.Order}

Apply the destriping algorithm to the TOD in `obs_list`. The result is
returned in `destriping_data`, which is of type
[`DestripingData`](@ref) and contains the baselines, I/Q/U maps,
weight map, etc.

The parameters must satisfy the following constraints:

- `obs_list` must be an array of `N` [`Observation`](@ref) objects;
- `destriping_data` must be a [`DestripingData`](@ref) object. It does
  not need to have been initialized using [`reset_maps!`](@ref).

The optional keyword have the following meaning and default value:

- `comm` is a MPI communicator object, or `nothing` is MPI is not
  used.
- `unseen` is the value to be used for pixels in the map that have not
  been observed. The default is `missing`; other sensible values are
  `nothing` and `NaN`.
- `callback` is either `nothing` (the default) or a function that is
  called before the CG algorithm starts, and then once every
  iteration. It can be used to monitor the progress of the destriper,
  or to save intermediate results in a database or a file. The
  function receives `destriping_data` as its parameter

The function uses Julia's `Logging` module. Therefore, you can enable
the output of diagnostic messages as usual; consider the following
example:

```julia
global_logger(ConsoleLogger(stderr, Debug))

# This command is going to produce a lot of output…
destripe!(...)
```

"""
function destripe!(
    obs_list::Vector{Observation{T}},
    destriping_data::DestripingData{T, O};
    comm = nothing,
    unseen = missing,
    callback = nothing,
) where {T <: Real, O <: Healpix.Order}

    @assert length(obs_list) == length(destriping_data.baselines)

    # This variable is not going to change during the execution of the function
    baseline_lengths = [runs(destriping_data.baselines[idx])
                        for idx in 1:length(obs_list)]

    r = [Array{T}(undef, length(values(destriping_data.baselines[idx])))
         for idx in 1:length(obs_list)]
    rnext = deepcopy(r)

    compute_nobs_matrix!(destriping_data.nobs_matrix, obs_list)

    compute_residuals!(
        r,
        baseline_lengths,
        destriping_data,
        obs_list,
        comm = comm,
        unseen = unseen,
    )
    k = 0

    destriping_data.stopping_factors = T[]

    stopping_factor = calc_stopping_factor(r, comm = comm)
    push!(destriping_data.stopping_factors, stopping_factor)
    stopping_factor < destriping_data.threshold && return

    precond = if destriping_data.use_preconditioner
        jacobi_preconditioner(
            [runs(b) for b in destriping_data.baselines],
            T
        )
    else
        IdentityPreconditioner()
    end

    z = deepcopy(r)
    apply!(precond, z)

    old_z_dot_r = array_dot(z, r, comm = comm)
    p = [RunLengthArray{Int, T}(runs(destriping_data.baselines[idx]), z[idx])
         for idx in 1:length(destriping_data.baselines)]
    best_stopping_factor = stopping_factor
    best_a = deepcopy(destriping_data.baselines)

    ap = [Array{T}(undef, length(r[idx])) for idx in 1:length(r)]
    while true
        k += 1
        @debug "CG iteration $k/$(destriping_data.max_iterations), stopping factor = $stopping_factor"

        if k >= destriping_data.max_iterations
            @warn "The destriper did not converge after $k iterations"
            destriping_data.baselines = best_a
            break
        end

        reset_maps!(destriping_data)
        compute_z_and_group!(
            ap,
            p,
            baseline_lengths,
            destriping_data,
            obs_list,
            comm = comm,
            unseen = unseen,
        )
        γ = old_z_dot_r / array_dot(p, ap, comm = comm)
        for obsidx in 1:length(obs_list)
            destriping_data.baselines[obsidx].values .+= γ * p[obsidx].values
            rnext[obsidx] .= r[obsidx] - γ * ap[obsidx]
        end
        # Remove the mean value from the baselines
        apply_offset_to_baselines!(destriping_data.baselines)

        stopping_factor = calc_stopping_factor(r, comm = comm)
        push!(destriping_data.stopping_factors, stopping_factor)
        if stopping_factor < best_stopping_factor
            best_a = deepcopy(destriping_data.baselines)
            best_stopping_factor = stopping_factor
        end
        if stopping_factor < destriping_data.threshold
            @info "Destriper converged after $k iterations, stopping factor ($(stopping_factor)) < threshold ($(destriping_data.threshold))"
            break
        end

        z = deepcopy(rnext)
        apply!(precond, z)
        new_z_dot_r = array_dot(z, rnext, comm = comm)

        for obsidx in 1:length(obs_list)
            r[obsidx] .= rnext[obsidx]
            p[obsidx].values .= z[obsidx] + (new_z_dot_r / old_z_dot_r) * p[obsidx].values
        end

        (callback === nothing) || callback(destriping_data)
        old_z_dot_r = new_z_dot_r
    end

    @debug "Producing the map"
    calculate_cleaned_map!(obs_list, destriping_data, comm = comm, unseen = unseen)
end
