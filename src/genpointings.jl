import Printf
import JSON
import Healpix
import LinearAlgebra: dot, ×
import Base: show

export MINUTES_PER_DAY,
       DAYS_PER_YEAR,
       rpm2angfreq,
       angfreq2rpm,
       period2rpm,
       rpm2period,
       ScanningStrategy,
       update_scanning_strategy,
       load_scanning_strategy,
       to_dict,
       save,
       time2pointing,
       time2pointing!,
       genpointings,
       genpointings!

################################################################################

rpm2angfreq(rpm) = 2π * rpm / 60
angfreq2rpm(ω) = ω / 2π * 60

@doc raw"""
    rpm2angfreq(rpm)
    angfreq2rpm(ω)

Convert rotations per minute into angular frequency 2πν (in Hertz), and vice
versa.
"""
rpm2angfreq, angfreq2rpm

period2rpm(p) = 60 / p
rpm2period(rpm) = 60 / rpm

@doc raw"""
    period2rpm(p)
    rpm2perriod(rpm)

Convert a period (time span) in seconds into a number of rotations per minute,
and vice versa.
"""
period2rpm, rpm2period

################################################################################

const MINUTES_PER_DAY = 60 * 24
const DAYS_PER_YEAR = 365.25

@doc raw"""
The structure `ScanningStrategy` encodes the information needed to build a
set of pointing directions. It contains the following fields:

- `omega_spin_hz`: angular speed of the rotation around the spin axis (equal to 2πν)
- `omega_prec_hz`: angular speed of the rotation around the precession axis (equal to 2πν)
- `omega_year_hz`: angular speed of the rotation around the Elliptic axis (equal to 2πν)
- `omega_hwp_hz`: angular speed of the rotation of the half-wave plate (equal to 2πν)
- `spinsunang_rad`: angle between the spin axis and the Sun-Earth direction
- `borespinang_rad`: angle between the boresight direction and the spin axis
- `q1`, `q3`: quaternions used to generate the pointings

Each field has its measure unit appended to the name. For instance, field
`spinsunang_rad` must be expressed in radians.

"""
struct ScanningStrategy
    omega_spin_hz::Float64
    omega_prec_hz::Float64
    omega_year_hz::Float64
    omega_hwp_hz::Float64
    spinsunang_rad::Float64
    borespinang_rad::Float64
    # First quaternion used in the rotation
    q1::Quaternion
    # Third quaternion used in the rotation
    q3::Quaternion
    
    ScanningStrategy(; spin_rpm = 0,
        prec_rpm = 0,
        yearly_rpm = 1  / (MINUTES_PER_DAY * DAYS_PER_YEAR),
        hwp_rpm = 0,
        spinsunang_rad = deg2rad(45.0),
        borespinang_rad = deg2rad(50.0),) = new(rpm2angfreq(spin_rpm),
        rpm2angfreq(prec_rpm),
        rpm2angfreq(yearly_rpm),
        rpm2angfreq(hwp_rpm),
        spinsunang_rad,
        borespinang_rad,
        compute_q1(borespinang_rad),
        compute_q3(spinsunang_rad))

    ScanningStrategy(io::IO) = load_scanning_strategy(io)
    ScanningStrategy(filename::AbstractString) = load_scanning_strategy(filename)
end

compute_q1(borespinang_rad) = qrotation_y(borespinang_rad)
compute_q3(spinsunang_rad) = qrotation_y(π / 2 - spinsunang_rad)

@doc raw"""
    update_scanning_strategy(sstr::ScanningStrategy)

Update the internal fields of a `ScanningStrategy` object. If you change any of the
fields in a `ScanningStrategy` object after it has been created using the constructors,
call this function
before using one of the functions `time2pointing`, `time2pointing!`, `genpointings`,
and `genpointings!`, as they rely on a number of internal parameters that need to be
updated.

```julia
sstr = ScanningStrategy()
# ... use sstr ...

sstr.borespinang_rad *= 0.5
update_scanning_strategy(sstr)
```

"""
function update_scanning_strategy(sstr::ScanningStrategy)
    sstr.q1 = compute_q1(sstr.borespinang_rad)
    sstr.q3 = compute_q3(sstr.spinsunang_rad)
end

################################################################################

function load_scanning_strategy(io::IO)
    data = JSON.Parser.parse(io)

    sstr_data = data["scanning_strategy"]

    ScanningStrategy(spin_rpm = sstr_data["spin_rpm"],
        prec_rpm = sstr_data["prec_rpm"],
        yearly_rpm = sstr_data["yearly_rpm"],
        hwp_rpm = sstr_data["hwp_rpm"],
        spinsunang_rad = sstr_data["spinsunang_rad"],
        borespinang_rad = sstr_data["borespinang_rad"])
end

function load_scanning_strategy(filename::AbstractString)
    open(filename) do inpf
        load_scanning_strategy(inpf)
    end
end

@doc raw"""
    load_scanning_strategy(io::IO) -> ScanningStrategy
    load_scanning_strategy(filename::AbstractString) -> ScanningStrategy

Create a `ScanningStrategy` object from the definition found in the JSON file
`io`, or from the JSON file with path `filename`. See also
[`load_scanning_strategy`](@ref).
"""
load_scanning_strategy

################################################################################

@doc raw"""
    to_dict(sstr::ScanningStrategy) -> Dict{String, Any}

Convert a scanning strategy into a dictionary suitable to be serialized using
JSON or any other structured format. See also [`save`](@ref).
"""
function to_dict(sstr::ScanningStrategy)
    Dict("scanning_strategy" => Dict("spin_rpm" => angfreq2rpm(sstr.omega_spin_hz),
        "prec_rpm" => angfreq2rpm(sstr.omega_prec_hz),
        "yearly_rpm" => angfreq2rpm(sstr.omega_year_hz),
        "hwp_rpm" => angfreq2rpm(sstr.omega_hwp_hz),
        "spinsunang_rad" => sstr.spinsunang_rad,
        "borespinang_rad" => sstr.borespinang_rad))
end

function save(io::IO, sstr::ScanningStrategy)
    print(io, JSON.json(to_dict(sstr), 4))
end

function save(filename::AbstractString, sstr::ScanningStrategy)
    open(filename, "w") do outf
        save(outf, sstr)
    end
end

@doc raw"""
    save(io::IO, sstr::ScanningStrategy)
    save(filename::AbstractString, sstr::ScanningStrategy)

Write a definition of the scanning strategy in a self-contained JSON file.
You can reload this definition using one of the constructors of
[`ScanningStrategy`](@ref).
"""
save

################################################################################

function show(io::IO, sstr::ScanningStrategy)
    Printf.@printf(io, """Scanning strategy:
    Spin angular velocity.................................... %g rot/s
    Precession angular velocity.............................. %g rot/s
    Yearly angular velocity around the Sun................... %g rot/s
    Half-wave plate angular velocity......................... %g rot/s
    Angle between the spin axis and the Sun-Earth direction.. %f°
    Angle between the boresight direction and the spin axis.. %f°
""",
        sstr.omega_spin_hz,
        sstr.omega_prec_hz,
        sstr.omega_year_hz,
        sstr.omega_hwp_hz,
        rad2deg(sstr.spinsunang_rad),
        rad2deg(sstr.borespinang_rad))
end

################################################################################

function time2pointing!(sstr::ScanningStrategy, time_s, beam_dir, polangle_rad, resultvec)
    curpolang = mod2pi(polangle_rad + 4 * sstr.omega_hwp_hz * time_s)
    # The polarization vector lies on the XY plane; if polangle_rad=0 then
    # the vector points along the X direction at t=0.
    poldir = StaticArrays.SVector(cos(curpolang), sin(curpolang), 0.0)
    
    q2 = qrotation_z(sstr.omega_spin_hz * time_s)
    q4 = qrotation_x(sstr.omega_prec_hz * time_s)
    q5 = qrotation_z(sstr.omega_year_hz * time_s)
    
    qtot = q5 * (q4 * (sstr.q3 * (q2 * sstr.q1)))
    rot = rotationmatrix_normalized(qtot)
    # Direction in the sky of the beam main axis
    resultvec[4:6] = rot * beam_dir
    # Direction in the sky of the beam polarization axis
    poldir = rot * poldir
    
    # The North for a vector v is just -dv/dθ, as θ is the
    # colatitude and moves along the meridian
    (θ, ϕ) = Healpix.vec2ang(resultvec[4:6]...)
    northdir = StaticArrays.SVector(-cos(θ) * cos(ϕ), -cos(θ) * sin(ϕ), sin(θ))
    
    cosψ = clamp(dot(northdir, poldir), -1, 1)
    crosspr = northdir × poldir
    sinψ = clamp(sqrt(dot(crosspr, crosspr)), -1, 1)
    resultvec[3] = atan(cosψ, sinψ)

    resultvec[1], resultvec[2] = θ, ϕ
end

function time2pointing(sstr::ScanningStrategy, time_s, beam_dir, polangle_rad)
    resultvec = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time2pointing!(sstr, time_s, beam_dir, polangle_rad, resultvec)
    resultvec
end

@doc raw"""
    time2pointing!(sstr::ScanningStrategy, time_s, beam_dir, polangle_rad, resultvec)
    time2pointing(sstr::ScanningStrategy, time_s, beam_dir, polangle_rad)

Calculate the pointing direction of a beam along the direction `beam_dir`, with
a detector sensitive to the polarization along the angle `polangle_rad`. The
result is saved in `resultvec` for `time2pointing!`, and it is the return value
of `time2pointing`; it is a 6-element array containing the following fields:

1. The colatitude (in radians) of the point in the sky
2. The longitude (in radians) of the point in the sky
3. The polarization angle, in the reference frame of the sky
4. The X component of the normalized pointing vector
5. The Y component
6. The Z component

Fields #4, #5, #6 are redundant, as they can be derived from the colatitude
(field #1) and longitude (field #2). They are returned as the code already
computes them.

The vector `beam_dir` and the angle `polangle_rad` must be expressed in the
reference system of the focal plane. If `polangle_rad == 0`, the detector
measures polarization along the x axis of the focal plane. The normal direction
to the focal plane is along the z axis; thus, the boresight director is such
that `beam_dir = [0., 0., 1.]`.

"""
time2pointing!, time2pointing

################################################################################

function genpointings!(sstr::ScanningStrategy, timerange_s, beam_dir, polangle_rad, result)
    @assert size(result)[1] == length(timerange_s)

    @inbounds for (idx, t) in enumerate(timerange_s)
        time2pointing!(sstr, t, beam_dir, polangle_rad, view(result, idx, :))
    end
end

function genpointings(sstr::ScanningStrategy, timerange_s, beam_dir, polangle_rad)
    result = Array{Float64}(undef, length(timerange_s), 6)
    genpointings!(sstr, timerange_s, beam_dir, polangle_rad, result)

    result
end

@doc raw"""
    genpointings!(sstr::ScanningStrategy, timerange_s, beam_dir, polangle_rad, result)
    genpointings(sstr::ScanningStrategy, timerange_s, beam_dir, polangle_rad)

Generate a set of pointing directions and angles for a given orientation
`beam_dir` (a 3-element vector) of the boresight beam, assuming the scanning
strategy in `sstr`. The pointing directions are calculated over all the elements
of the list `timerange_s`.

The two functions only differ in the way the result is returned to the caller.
Function `genpointings` returns a N×6 matrix containing the following fields:

1. The colatitude (in radians)
2. The longitude (in radians)
3. The polarization angle (in radians)
4. The X component of the one-length pointing vector
5. The Y component
6. The Z component

Function `genpointings!` works like `genpointings`, but it accept a
pre-allocated matrix as input (the `result` parameter) and will save the result
in it. The matrix must have two dimensions with size `(N, 6)` at least.

Both functions are simple iterators wrapping [`time2pointing!`](@ref) and
[`time2pointing`](@ref).

"""
genpointings!, genpointings
