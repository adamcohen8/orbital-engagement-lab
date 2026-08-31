import java.io.File;
import java.util.Locale;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.data.LazyLoadedDataContext;
import org.orekit.files.ccsds.ndm.odm.oem.Oem;
import org.orekit.frames.EOPHistory;
import org.orekit.frames.Frame;
import org.orekit.frames.PoleCorrection;
import org.orekit.frames.Transform;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScales;
import org.orekit.utils.CartesianDerivativesFilter;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;

public final class OrekitFrameTimeAcceptance {
    private static final double ARCSEC_PER_RAD = 206264.80624709636;

    private OrekitFrameTimeAcceptance() {
    }

    public static void main(final String[] args) {
        if (args.length != 1) {
            throw new IllegalArgumentException("usage: OrekitFrameTimeAcceptance <orekit-data-path>");
        }
        Locale.setDefault(Locale.ROOT);
        final LazyLoadedDataContext context = DataContext.getDefault();
        context.getDataProvidersManager().clearProviders();
        context.getDataProvidersManager().addProvider(new DirectoryCrawler(new File(args[0])));
        final TimeScales timeScales = context.getTimeScales();
        final TimeScale utc = timeScales.getUTC();
        final TimeScale tai = timeScales.getTAI();
        final TimeScale tt = timeScales.getTT();
        final AbsoluteDate beforeLeap = new AbsoluteDate("2016-12-31T23:59:59", utc);
        final AbsoluteDate leap = beforeLeap.shiftedBy(1.0);
        final AbsoluteDate afterLeap = new AbsoluteDate("2017-01-01T00:00:00", utc);
        final AbsoluteDate epoch = new AbsoluteDate("2024-01-01T00:00:00", utc);
        final AbsoluteDate current = new AbsoluteDate("2026-08-29T12:00:00", utc);

        final EOPHistory eop = context.getFrames().getEOPHistory(IERSConventions.IERS_1996, false);
        final PoleCorrection pole = eop.getPoleCorrection(epoch);
        final double[] nutation = eop.getEquinoxNutationCorrection(epoch);

        final Frame eme2000 = context.getFrames().getEME2000();
        final Frame itrf = context.getFrames().getITRF(IERSConventions.IERS_1996, false);
        final Frame gcrf = context.getFrames().getGCRF();
        final Frame itrf2010 = context.getFrames().getITRF(IERSConventions.IERS_2010, false);
        final Frame teme = context.getFrames().getTEME();
        final PVCoordinates pv = new PVCoordinates(
                new Vector3D(7000000.0, 120000.0, 30000.0),
                new Vector3D(-200.0, 7450.0, 1100.0));

        final Transform emeToItrf = eme2000.getTransformTo(itrf, epoch);
        final PVCoordinates itrfPv = emeToItrf.transformPVCoordinates(pv);
        final double[][] emeToItrfJacobian = new double[6][6];
        emeToItrf.getJacobian(CartesianDerivativesFilter.USE_PV, emeToItrfJacobian);

        final Transform temeToEme = teme.getTransformTo(eme2000, epoch);
        final PVCoordinates emePv = temeToEme.transformPVCoordinates(pv);

        final Transform gcrfToItrf = gcrf.getTransformTo(itrf2010, epoch);
        final PVCoordinates gcrfItrfPv = gcrfToItrf.transformPVCoordinates(pv);
        final double[][] gcrfToItrfJacobian = new double[6][6];
        gcrfToItrf.getJacobian(CartesianDerivativesFilter.USE_PV, gcrfToItrfJacobian);
        final Transform gcrfToEme = gcrf.getTransformTo(eme2000, epoch);
        final PVCoordinates gcrfEmePv = gcrfToEme.transformPVCoordinates(pv);

        final String version = Oem.class.getProtectionDomain().getCodeSource().getLocation().getPath();
        print("orekit_version", version.replaceFirst("^.*/orekit-", "").replaceFirst("\\.jar$", ""));
        print("utc_before", beforeLeap.toStringWithoutUtcOffset(utc, 6));
        print("utc_leap", leap.toStringWithoutUtcOffset(utc, 6));
        print("utc_after", afterLeap.toStringWithoutUtcOffset(utc, 6));
        print("seconds_before_to_leap", leap.durationFrom(beforeLeap));
        print("seconds_leap_to_after", afterLeap.durationFrom(leap));
        print("utc_minus_tai_2026_s", current.timeScalesOffset(utc, tai));
        print("tt_minus_tai_2026_s", current.timeScalesOffset(tt, tai));
        print("dut1_s", eop.getUT1MinusUTC(epoch));
        print("xp_arcsec", pole.getXp() * ARCSEC_PER_RAD);
        print("yp_arcsec", pole.getYp() * ARCSEC_PER_RAD);
        print("ddpsi_rad", nutation[0]);
        print("ddeps_rad", nutation[1]);
        printVector("eme2000_to_itrf_position_m", itrfPv.getPosition());
        printVector("eme2000_to_itrf_velocity_m_s", itrfPv.getVelocity());
        printMatrix("eme2000_to_itrf_jacobian", emeToItrfJacobian);
        printVector("teme_to_eme2000_position_m", emePv.getPosition());
        printVector("teme_to_eme2000_velocity_m_s", emePv.getVelocity());
        printVector("gcrf_to_itrf_position_m", gcrfItrfPv.getPosition());
        printVector("gcrf_to_itrf_velocity_m_s", gcrfItrfPv.getVelocity());
        printMatrix("gcrf_to_itrf_jacobian", gcrfToItrfJacobian);
        printVector("gcrf_to_eme2000_position_m", gcrfEmePv.getPosition());
        printVector("gcrf_to_eme2000_velocity_m_s", gcrfEmePv.getVelocity());
    }

    private static void print(final String key, final String value) {
        System.out.printf(Locale.ROOT, "%s=%s%n", key, value);
    }

    private static void print(final String key, final double value) {
        System.out.printf(Locale.ROOT, "%s=%.17g%n", key, value);
    }

    private static void printVector(final String key, final Vector3D value) {
        System.out.printf(
                Locale.ROOT,
                "%s=%.17g,%.17g,%.17g%n",
                key,
                value.getX(),
                value.getY(),
                value.getZ());
    }

    private static void printMatrix(final String key, final double[][] value) {
        System.out.printf(Locale.ROOT, "%s=", key);
        for (int row = 0; row < value.length; row++) {
            for (int column = 0; column < value[row].length; column++) {
                if (row != 0 || column != 0) {
                    System.out.print(",");
                }
                System.out.printf(Locale.ROOT, "%.17g", value[row][column]);
            }
        }
        System.out.println();
    }
}
