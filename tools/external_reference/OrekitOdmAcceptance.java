import java.io.File;
import java.util.Locale;

import org.orekit.data.DataContext;
import org.orekit.data.DataSource;
import org.orekit.data.DirectoryCrawler;
import org.orekit.data.LazyLoadedDataContext;
import org.orekit.files.ccsds.ndm.ParserBuilder;
import org.orekit.files.ccsds.ndm.odm.omm.Omm;
import org.orekit.files.ccsds.ndm.odm.opm.Opm;
import org.orekit.utils.PVCoordinates;

public final class OrekitOdmAcceptance {
    private OrekitOdmAcceptance() {
    }

    public static void main(final String[] args) {
        if (args.length != 3) {
            throw new IllegalArgumentException("usage: OrekitOdmAcceptance <orekit-data> <opm> <omm>");
        }
        Locale.setDefault(Locale.ROOT);
        final LazyLoadedDataContext context = DataContext.getDefault();
        context.getDataProvidersManager().clearProviders();
        context.getDataProvidersManager().addProvider(new DirectoryCrawler(new File(args[0])));
        final ParserBuilder builder = new ParserBuilder(context);
        final Opm opm = builder.buildOpmParser().parseMessage(new DataSource(new File(args[1])));
        final Omm omm = builder.buildOmmParser().parseMessage(new DataSource(new File(args[2])));
        final PVCoordinates pv = opm.getPVCoordinates();

        print("opm_version", opm.getHeader().getFormatVersion());
        print("opm_object_id", opm.getMetadata().getObjectID());
        print("opm_frame", opm.getMetadata().getReferenceFrame().getName());
        print("opm_maneuver_count", opm.getNbManeuvers());
        printVector("opm_position_m", pv.getPosition().toArray());
        printVector("opm_velocity_m_s", pv.getVelocity().toArray());
        print("opm_covariance_present", opm.getData().getCovarianceBlock() == null ? "false" : "true");

        print("omm_version", omm.getHeader().getFormatVersion());
        print("omm_object_id", omm.getMetadata().getObjectID());
        print("omm_frame", omm.getMetadata().getReferenceFrame().getName());
        print("omm_theory", omm.getMetadata().getMeanElementTheory());
        print("omm_mean_motion_rad_s", omm.getData().getKeplerianElementsBlock().getMeanMotion());
        print("omm_eccentricity", omm.getData().getKeplerianElementsBlock().getE());
        print("omm_inclination_rad", omm.getData().getKeplerianElementsBlock().getI());
        print("omm_norad_id", omm.getData().getTLEBlock().getNoradID());
    }

    private static void print(final String key, final String value) {
        System.out.printf(Locale.ROOT, "%s=%s%n", key, value);
    }

    private static void print(final String key, final int value) {
        System.out.printf(Locale.ROOT, "%s=%d%n", key, value);
    }

    private static void print(final String key, final double value) {
        System.out.printf(Locale.ROOT, "%s=%.17g%n", key, value);
    }

    private static void printVector(final String key, final double[] value) {
        System.out.printf(Locale.ROOT, "%s=%.17g,%.17g,%.17g%n", key, value[0], value[1], value[2]);
    }
}
