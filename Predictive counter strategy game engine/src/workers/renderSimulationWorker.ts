// Dedicated Render Worker for OffscreenCanvas Dynamic Spatial Influence & Potential Fields
export interface SpatialUnit {
  row: number;
  col: number;
  weight: number;
  decayLambda: number;
  rangeHorizon: number;
  mobilityBeta: number;
  attackVectorRow: number;
  attackVectorCol: number;
  isFriendly: boolean;
}

/**
 * Computes the net Directionally Biased Dynamic Influence tensor M(r, c)
 * I_u(r, c) = W_u * exp(-lambda_u * d(u, r, c)) * (1 + beta_u * cos(phi_{u -> (r,c)}))
 */
export function computeDynamicInfluenceField(
  rows: number,
  cols: number,
  units: SpatialUnit[]
): Float32Array {
  const field = new Float32Array(rows * cols);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      let netInfluence = 0;

      for (const u of units) {
        const dr = r - u.row;
        const dc = c - u.col;
        const dist = Math.hypot(dr, dc);
        if (dist > u.rangeHorizon) continue;

        let cosPhi = 0;
        const vMag = Math.hypot(u.attackVectorRow, u.attackVectorCol);
        if (dist > 1e-5 && vMag > 1e-5) {
          cosPhi = (dr * u.attackVectorRow + dc * u.attackVectorCol) / (dist * vMag);
        }

        const mag = u.weight * Math.exp(-u.decayLambda * dist) * (1.0 + u.mobilityBeta * cosPhi);
        netInfluence += u.isFriendly ? mag : -mag;
      }

      field[r * cols + c] = netInfluence;
    }
  }
  return field;
}

/**
 * Co-evolved Attractive-Repulsive Potential Field equation for tactical flanking & micro
 * PF(d) = c_attr * d^{e_attr} - c_rep * d^{e_rep}
 */
export function evaluatePotentialFieldForce(
  distance: number,
  cAttr = 1.45,
  eAttr = -0.85,
  cRep = 2.10,
  eRep = -1.65
): number {
  const safeDist = Math.max(0.25, distance);
  return cAttr * Math.pow(safeDist, eAttr) - cRep * Math.pow(safeDist, eRep);
}