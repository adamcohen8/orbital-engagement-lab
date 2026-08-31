import java.io.File;
import java.util.List;
import java.util.Locale;

import org.orekit.data.DataContext;
import org.orekit.data.DataSource;
import org.orekit.data.DirectoryCrawler;
import org.orekit.data.LazyLoadedDataContext;
import org.orekit.files.ccsds.ndm.ParserBuilder;
import org.orekit.files.ccsds.ndm.odm.CartesianCovariance;
import org.orekit.files.ccsds.ndm.odm.oem.Oem;
import org.orekit.files.ccsds.ndm.odm.oem.OemSegment;
import org.orekit.utils.TimeStampedPVCoordinates;

public final class OrekitOemAcceptance {
    private OrekitOemAcceptance() {
    }

    public static void main(final String[] args) {
        if (args.length != 3) {
            throw new IllegalArgumentException(
                    "usage: OrekitOemAcceptance <oem-path> <orekit-data-path> <center-mu-m3-s2>");
        }
        Locale.setDefault(Locale.ROOT);
        final LazyLoadedDataContext context = DataContext.getDefault();
        context.getDataProvidersManager().clearProviders();
        context.getDataProvidersManager().addProvider(new DirectoryCrawler(new File(args[1])));
        final Oem oem = new ParserBuilder(context)
                .withMu(Double.parseDouble(args[2]))
                .buildOemParser()
                .parse(new DataSource(new File(args[0])));
        final List<OemSegment> segments = oem.getSegments();
        int stateCount = 0;
        int covarianceCount = 0;
        for (final OemSegment segment : segments) {
            stateCount += segment.getCoordinates().size();
            covarianceCount += segment.getCovarianceMatrices().size();
        }
        final OemSegment firstSegment = segments.get(0);
        final List<TimeStampedPVCoordinates> coordinates = firstSegment.getCoordinates();
        final TimeStampedPVCoordinates first = coordinates.get(0);
        final TimeStampedPVCoordinates last = coordinates.get(coordinates.size() - 1);
        final String version = Oem.class.getProtectionDomain().getCodeSource().getLocation().getPath();

        System.out.printf(
                "orekit_version=%s%n",
                version.replaceFirst("^.*/orekit-", "").replaceFirst("\\.jar$", ""));
        System.out.printf("segment_count=%d%n", segments.size());
        System.out.printf("state_count=%d%n", stateCount);
        System.out.printf("covariance_count=%d%n", covarianceCount);
        System.out.printf("object_name=%s%n", firstSegment.getMetadata().getObjectName());
        System.out.printf("object_id=%s%n", firstSegment.getMetadata().getObjectID());
        System.out.printf("center_name=%s%n", firstSegment.getMetadata().getCenter().getName());
        System.out.printf("ref_frame=%s%n", firstSegment.getMetadata().getReferenceFrame().getName());
        System.out.printf("time_system=%s%n", firstSegment.getMetadata().getTimeSystem());
        System.out.printf("first_x_m=%.17g%n", first.getPosition().getX());
        System.out.printf("first_vy_m_s=%.17g%n", first.getVelocity().getY());
        System.out.printf("last_x_m=%.17g%n", last.getPosition().getX());
        System.out.printf("last_vy_m_s=%.17g%n", last.getVelocity().getY());
        if (covarianceCount > 0) {
            final CartesianCovariance covariance = firstSegment.getCovarianceMatrices().get(0);
            System.out.printf("first_cov_ref_frame=%s%n", covariance.getReferenceFrame().getName());
            System.out.printf("first_cov_00_si=%.17g%n", covariance.getCovarianceMatrix().getEntry(0, 0));
            System.out.printf("first_cov_10_si=%.17g%n", covariance.getCovarianceMatrix().getEntry(1, 0));
            System.out.printf("first_cov_55_si=%.17g%n", covariance.getCovarianceMatrix().getEntry(5, 5));
        }
    }
}
