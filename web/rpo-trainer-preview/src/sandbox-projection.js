import {
  ellipticLinearCoastStates,
  keplerianToEci,
  stepTwoBodyState,
} from "./competition/arcade-engine.js";
import { sandboxRelativeSeed, sandboxTargetCoes } from "./sandbox-setup.js";

export function sandboxTargetStateAt(setup, timeS = 0, muKm3S2 = 398600.4418) {
  let target = keplerianToEci(sandboxTargetCoes(setup), muKm3S2);
  let remainingS = Math.max(Number(timeS || 0), 0);
  while (remainingS > 1.0e-9) {
    const dtS = Math.min(remainingS, 60);
    target = stepTwoBodyState(target, muKm3S2, dtS);
    remainingS -= dtS;
  }
  return target;
}

export function sandboxEllipticLinearCoastStates(setup, seed = {}, timesS = [], muKm3S2 = 398600.4418) {
  const startTimeS = Math.max(Number(seed.t || 0), 0);
  const chief = sandboxTargetStateAt(setup, startTimeS, muKm3S2);
  const defaults = sandboxRelativeSeed(setup);
  const relativeSeed = {
    r: Number(seed.r ?? defaults.r),
    i: Number(seed.i ?? defaults.i),
    c: Number(seed.c ?? defaults.c),
    rd: Number(seed.rd ?? defaults.rd),
    id: Number(seed.id ?? defaults.id),
    cd: Number(seed.cd ?? defaults.cd),
  };
  return ellipticLinearCoastStates(
    {
      r_km: relativeSeed.r,
      i_km: relativeSeed.i,
      c_km: relativeSeed.c,
      rd_km_s: relativeSeed.rd,
      id_km_s: relativeSeed.id,
      cd_km_s: relativeSeed.cd,
    },
    timesS,
    chief,
    muKm3S2,
  ).map((point) => ({
    r: Number(point.r || 0),
    i: Number(point.i || 0),
    c: Number(point.c || 0),
    rd: Number(point.rd || 0),
    id: Number(point.id || 0),
    cd: Number(point.cd || 0),
    t: startTimeS + Number(point.t || 0),
    dv: Number(seed.dv || 0),
  }));
}
