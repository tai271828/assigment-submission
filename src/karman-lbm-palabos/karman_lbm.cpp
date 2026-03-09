/**
 * Kármán Vortex Street using Palabos LBM framework.
 *
 * Geometry: Schäfer-Turek benchmark
 *   - Channel: 2.2 m x 0.41 m  (lx/ly = 2.2/0.41 ≈ 5.37)
 *   - Cylinder center: (0.2 m, 0.2 m), diameter D = 0.1 m
 *   - Parabolic inlet (left), zero-gradient outlet (right)
 *   - No-slip top/bottom walls
 *
 * The cylinder is slightly off-center (cy = 0.2/0.41 ≈ 0.488) to trigger
 * the asymmetric vortex shedding characteristic of the Kármán vortex street.
 *
 * Usage:
 *   ./karman_lbm [Re] [N]
 *   Re: Reynolds number (default: 100)
 *   N:  Resolution - lattice nodes across channel height (default: 82)
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "palabos2D.h"
#include "palabos2D.hh"

using namespace plb;
using namespace plb::descriptors;
using namespace std;

typedef double T;
#define DESCRIPTOR D2Q9Descriptor

// --- Physical geometry (Schäfer-Turek benchmark) ---
const T LX_PHYS = 2.2;    // channel length [m]
const T LY_PHYS = 0.41;   // channel height [m]
const T CX_PHYS = 0.2;    // cylinder center x [m]
const T CY_PHYS = 0.2;    // cylinder center y [m]
const T D_PHYS  = 0.1;    // cylinder diameter [m]

// --- Poiseuille helpers ---

T poiseuilleVelocity(plint iY, IncomprFlowParam<T> const &parameters)
{
    T y = (T)iY / parameters.getResolution();
    return 4. * parameters.getLatticeU() * (y - y * y);
}

T poiseuillePressure(plint iX, IncomprFlowParam<T> const &parameters)
{
    T Lx = parameters.getNx() - 1;
    T Ly = parameters.getNy() - 1;
    return 8. * parameters.getLatticeNu() * parameters.getLatticeU() / (Ly * Ly)
           * (Lx / (T)2 - (T)iX);
}

T poiseuilleDensity(plint iX, IncomprFlowParam<T> const &parameters)
{
    return poiseuillePressure(iX, parameters) * DESCRIPTOR<T>::invCs2 + (T)1;
}

// --- Functionals ---

template <typename T_>
class PoiseuilleVelocity {
public:
    PoiseuilleVelocity(IncomprFlowParam<T_> parameters_) : parameters(parameters_) { }
    void operator()(plint, plint iY, Array<T_, 2> &u) const
    {
        u[0] = poiseuilleVelocity(iY, parameters);
        u[1] = T_();
    }
private:
    IncomprFlowParam<T_> parameters;
};

template <typename T_>
class ConstantDensity {
public:
    ConstantDensity(T_ density_) : density(density_) { }
    T_ operator()(plint, plint) const { return density; }
private:
    T_ density;
};

template <typename T_>
class PoiseuilleVelocityAndDensity {
public:
    PoiseuilleVelocityAndDensity(IncomprFlowParam<T_> parameters_) : parameters(parameters_) { }
    void operator()(plint iX, plint iY, T_ &rho, Array<T_, 2> &u) const
    {
        rho = poiseuilleDensity(iX, parameters);
        u[0] = poiseuilleVelocity(iY, parameters);
        u[1] = T_();
    }
private:
    IncomprFlowParam<T_> parameters;
};

// --- Cylinder geometry ---

template <typename T_>
class CylinderShapeDomain2D : public plb::DomainFunctional2D {
public:
    CylinderShapeDomain2D(plb::plint cx_, plb::plint cy_, plb::plint radius_) :
        cx(cx_), cy(cy_), radiusSqr(plb::util::sqr(radius_))
    { }
    virtual bool operator()(plb::plint iX, plb::plint iY) const
    {
        return plb::util::sqr(iX - cx) + plb::util::sqr(iY - cy) <= radiusSqr;
    }
    virtual CylinderShapeDomain2D<T_> *clone() const
    {
        return new CylinderShapeDomain2D<T_>(*this);
    }
private:
    plb::plint cx;
    plb::plint cy;
    plb::plint radiusSqr;
};

// --- Setup ---

void cylinderSetup(
    MultiBlockLattice2D<T, DESCRIPTOR> &lattice,
    IncomprFlowParam<T> const &parameters,
    OnLatticeBoundaryCondition2D<T, DESCRIPTOR> &boundaryCondition,
    plint cx_lb, plint cy_lb, plint radius_lb)
{
    const plint nx = parameters.getNx();
    const plint ny = parameters.getNy();
    Box2D outlet(nx - 1, nx - 1, 1, ny - 2);

    // Velocity BCs on left, top, bottom
    boundaryCondition.setVelocityConditionOnBlockBoundaries(
        lattice, Box2D(0, 0, 1, ny - 2));           // inlet
    boundaryCondition.setVelocityConditionOnBlockBoundaries(
        lattice, Box2D(0, nx - 1, 0, 0));            // bottom wall
    boundaryCondition.setVelocityConditionOnBlockBoundaries(
        lattice, Box2D(0, nx - 1, ny - 1, ny - 1));  // top wall
    // Outflow on right
    boundaryCondition.setVelocityConditionOnBlockBoundaries(
        lattice, Box2D(nx - 1, nx - 1, 1, ny - 2), boundary::outflow);

    setBoundaryVelocity(
        lattice, lattice.getBoundingBox(), PoiseuilleVelocity<T>(parameters));
    setBoundaryDensity(lattice, outlet, ConstantDensity<T>(1.));
    initializeAtEquilibrium(
        lattice, lattice.getBoundingBox(), PoiseuilleVelocityAndDensity<T>(parameters));

    // Place cylinder with bounce-back
    defineDynamics(
        lattice, lattice.getBoundingBox(),
        new CylinderShapeDomain2D<T>(cx_lb, cy_lb, radius_lb),
        new plb::BounceBack<T, DESCRIPTOR>);

    lattice.initialize();
}

// --- Output ---

void writeGif(MultiBlockLattice2D<T, DESCRIPTOR> &lattice, plint iter)
{
    ImageWriter<T> imageWriter("leeloo");
    imageWriter.writeScaledGif(
        createFileName("u", iter, 6),
        *computeVelocityNorm(lattice));
}

void writeVTK(
    MultiBlockLattice2D<T, DESCRIPTOR> &lattice,
    IncomprFlowParam<T> const &parameters, plint iter)
{
    T dx = parameters.getDeltaX();
    T dt = parameters.getDeltaT();
    VtkImageOutput2D<T> vtkOut(createFileName("vtk", iter, 6), dx);
    vtkOut.writeData<float>(
        *computeVelocityNorm(lattice), "velocityNorm", dx / dt);
    vtkOut.writeData<2, float>(
        *computeVelocity(lattice), "velocity", dx / dt);
}

// --- Main ---

int main(int argc, char *argv[])
{
    plbInit(&argc, &argv);

    // Parse command-line arguments:
    //   Re_D: Reynolds number based on cylinder diameter (as in assignment)
    //   N:    lattice nodes across channel height
    T Re_D = 100.;
    plint N = 82;

    if (argc > 1) Re_D = atof(argv[1]);
    if (argc > 2) N = atoi(argv[2]);

    // Compute lattice geometry from physical dimensions
    T lx_over_ly = LX_PHYS / LY_PHYS;  // aspect ratio ≈ 5.37
    T uMax = 0.02;                       // peak lattice velocity (low Mach)

    // Convert Re_D (cylinder-diameter-based) to Re_H (channel-height-based)
    // because Palabos IncomprFlowParam uses Re_H = uMax * H / nu.
    // Since nu is the same: Re_H = Re_D * H / D = Re_D * ly / D_phys
    T Re_H = Re_D * (LY_PHYS / D_PHYS);  // = Re_D * 4.1

    pcout << "============================================" << endl;
    pcout << "Karman Vortex Street - Palabos LBM" << endl;
    pcout << "============================================" << endl;
    pcout << "Re_D (cylinder-based) = " << Re_D << endl;
    pcout << "Re_H (channel-based)  = " << Re_H << endl;
    pcout << "N  = " << N << " (lattice nodes across channel height)" << endl;

    // Set output directory based on Re_D
    std::ostringstream outDir;
    outDir << "./tmp_Re" << (int)Re_D << "/";
    global::directories().setOutputDir(outDir.str());

    IncomprFlowParam<T> parameters(
        uMax,            // uMax (peak lattice velocity)
        Re_H,            // Re (channel-height-based, as Palabos expects)
        N,               // N (resolution = lattice nodes across ly)
        lx_over_ly,      // lx (in units of ly)
        1.               // ly (reference length)
    );

    // Cylinder position in lattice units
    // Physical: cx=0.2m, cy=0.2m in a 2.2m x 0.41m channel
    // Ensure at least 2 nodes off-center to trigger vortex shedding.
    T dx = parameters.getDeltaX();  // = 1/N
    plint ny = parameters.getNy();
    plint cx_lb = (plint)(CX_PHYS / LY_PHYS * N);
    plint cy_lb = (plint)(CY_PHYS / LY_PHYS * N);
    plint radius_lb = (plint)(D_PHYS / 2.0 / LY_PHYS * N);
    plint D_lb = 2 * radius_lb;

    // Ensure at least 2-node offset from channel center
    plint cy_center = ny / 2;
    if (std::abs(cy_lb - cy_center) < 2) {
        cy_lb = cy_center + 2;
        pcout << "NOTE: Adjusted cy to " << cy_lb
              << " (offset from center " << cy_center
              << ") to trigger vortex shedding" << endl;
    }

    // Verify Re_D matches expectation
    T nu_lb = parameters.getLatticeNu();
    T Re_D_actual = uMax * D_lb / nu_lb;

    pcout << "Grid: " << parameters.getNx() << " x " << ny << endl;
    pcout << "Cylinder: center=(" << cx_lb << "," << cy_lb
          << "), radius=" << radius_lb << ", D_lb=" << D_lb << endl;
    pcout << "tau = " << 1. / parameters.getOmega() << endl;
    pcout << "nu_lb = " << nu_lb << endl;
    pcout << "uMax_lb = " << parameters.getLatticeU() << endl;
    pcout << "Re_D (actual) = " << Re_D_actual << endl;
    pcout << "dx = " << dx << ", dt = " << parameters.getDeltaT() << endl;
    pcout << "============================================" << endl;

    // Timing parameters (in physical time units)
    const T logT   = (T)0.1;    // log interval
    const T imSave = (T)0.5;    // gif save interval
    const T vtkSave = (T)2.0;   // vtk save interval
    const T maxT   = (T)30.0;   // total simulation time

    writeLogFile(parameters, "Karman Vortex Street");

    // Create lattice with BGK dynamics
    MultiBlockLattice2D<T, DESCRIPTOR> lattice(
        parameters.getNx(), parameters.getNy(),
        new BGKdynamics<T, DESCRIPTOR>(parameters.getOmega()));

    OnLatticeBoundaryCondition2D<T, DESCRIPTOR> *boundaryCondition =
        createLocalBoundaryCondition2D<T, DESCRIPTOR>();

    cylinderSetup(lattice, parameters, *boundaryCondition,
                  cx_lb, cy_lb, radius_lb);

    pcout << "Starting simulation..." << endl;

    // Main time loop
    for (plint iT = 0; iT * parameters.getDeltaT() < maxT; ++iT) {
        if (iT % parameters.nStep(imSave) == 0) {
            pcout << "Saving Gif at t=" << iT * parameters.getDeltaT() << endl;
            writeGif(lattice, iT);
        }

        if (iT % parameters.nStep(vtkSave) == 0 && iT > 0) {
            pcout << "Saving VTK at t=" << iT * parameters.getDeltaT() << endl;
            writeVTK(lattice, parameters, iT);
        }

        if (iT % parameters.nStep(logT) == 0) {
            pcout << "step " << iT
                  << "; t=" << iT * parameters.getDeltaT()
                  << "; av energy=" << setprecision(10)
                  << getStoredAverageEnergy<T>(lattice)
                  << "; av rho=" << getStoredAverageDensity<T>(lattice)
                  << endl;
        }

        // LBM step
        lattice.collideAndStream();
    }

    pcout << "Simulation complete." << endl;

    delete boundaryCondition;
    return 0;
}
