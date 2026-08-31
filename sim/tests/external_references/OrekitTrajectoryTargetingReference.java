import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Locale;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.Orbit;
import org.orekit.propagation.Propagator;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.PVCoordinates;

/** Independent, deterministic Orekit references for OEL public trajectory targeting. */
public final class OrekitTrajectoryTargetingReference {
    private static final double MU_M3_S2 = 3.986004418e14;
    private static final Frame FRAME = FramesFactory.getGCRF();

    private OrekitTrajectoryTargetingReference() { }

    public static void main(final String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Usage: OrekitTrajectoryTargetingReference <output.csv>");
        }
        Locale.setDefault(Locale.ROOT);
        try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(args[0])))) {
            out.println("case_id,time_s,x_km,y_km,z_km,vx_km_s,vy_km_s,vz_km_s");
            writeHohmann(out);
            writeRendezvousCases(out);
        }
    }

    private static void writeHohmann(final PrintWriter out) {
        final double r1M = 7000000.0;
        final double r2M = 9000000.0;
        final double transferAM = 0.5 * (r1M + r2M);
        final double circularSpeed = Math.sqrt(MU_M3_S2 / r1M);
        final double transferSpeed = Math.sqrt(MU_M3_S2 * (2.0 / r1M - 1.0 / transferAM));
        final double transferTime = Math.PI * Math.sqrt(transferAM * transferAM * transferAM / MU_M3_S2);
        writeCase(out, "hohmann_half_transfer", transferTime,
                new double[] {r1M, 0.0, 0.0, 0.0, transferSpeed, 0.0});
        if (!(transferSpeed > circularSpeed)) {
            throw new IllegalStateException("Hohmann departure speed must exceed circular speed.");
        }
    }

    private static void writeRendezvousCases(final PrintWriter out) {
        final double rM = 7000000.0;
        final double circularSpeed = Math.sqrt(MU_M3_S2 / rM);
        final double[] base = {rM, 0.0, 0.0, 12.0, circularSpeed - 6.0, 2.0};
        final double perturbationMps = 0.1;
        writeCase(out, "rendezvous_seed", 900.0, base);
        final double[] plusX = base.clone();
        plusX[3] += perturbationMps;
        writeCase(out, "rendezvous_plus_x", 900.0, plusX);
        final double[] minusX = base.clone();
        minusX[3] -= perturbationMps;
        writeCase(out, "rendezvous_minus_x", 900.0, minusX);
        final double[] plusY = base.clone();
        plusY[4] += perturbationMps;
        writeCase(out, "rendezvous_plus_y", 900.0, plusY);
        final double[] minusY = base.clone();
        minusY[4] -= perturbationMps;
        writeCase(out, "rendezvous_minus_y", 900.0, minusY);
    }

    private static void writeCase(
            final PrintWriter out,
            final String caseId,
            final double durationS,
            final double[] stateMAndMps) {
        final AbsoluteDate epoch = AbsoluteDate.J2000_EPOCH;
        final Vector3D position = new Vector3D(stateMAndMps[0], stateMAndMps[1], stateMAndMps[2]);
        final Vector3D velocity = new Vector3D(stateMAndMps[3], stateMAndMps[4], stateMAndMps[5]);
        final Orbit orbit = new CartesianOrbit(new PVCoordinates(position, velocity), FRAME, epoch, MU_M3_S2);
        final Propagator propagator = new KeplerianPropagator(orbit);
        final PVCoordinates result = propagator.propagate(epoch.shiftedBy(durationS)).getPVCoordinates();
        out.printf(Locale.ROOT, "%s,%.12g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g%n",
                caseId,
                durationS,
                result.getPosition().getX() / 1000.0,
                result.getPosition().getY() / 1000.0,
                result.getPosition().getZ() / 1000.0,
                result.getVelocity().getX() / 1000.0,
                result.getVelocity().getY() / 1000.0,
                result.getVelocity().getZ() / 1000.0);
    }
}
