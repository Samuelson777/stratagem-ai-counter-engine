// Dedicated AI Worker: Chessformer (Maia-3) + Bayesian Profiler + OM-SO-ISMCTS GenBR
export interface ArchetypeParameter {
  id: string;
  elo: number;
  aggression: number;
  riskTolerance: number;
  posterior: number;
}

export interface ConceptProbeResult {
  conceptId: string;
  conceptName: string;
  masterActivation: number;
  opponentActivation: number;
  isExploitableBlindSpot: boolean;
}

/**
 * 1. Recursive Bayesian Belief Update with Bayes-Relational Prior Adaptation
 * B_{t+1}(theta_i) = [P(o_t | theta_i, s_t, h_t) * B_t(theta_i)] / sum_j(...)
 */
export function updateRecursiveBayesianBelief(
  portfolio: ArchetypeParameter[],
  moveLikelihoods: number[],
  relationalClassifierProbDpGivenX: number
): ArchetypeParameter[] {
  // Bayes-Relational adaptation ratio: P(x | D_p) = P(x | D) * [P(D_p | x) / (1 - P(D_p | x))]
  const clampedRel = Math.min(0.95, Math.max(0.05, relationalClassifierProbDpGivenX));
  const relationalRatio = clampedRel / (1.0 - clampedRel);

  let normalizer = 0;
  const unnormalized = portfolio.map((arch, idx) => {
    const adaptedLikelihood = moveLikelihoods[idx] * (idx === 0 ? relationalRatio : 1.0);
    const val = adaptedLikelihood * arch.posterior;
    normalizer += val;
    return val;
  });

  return portfolio.map((arch, idx) => ({
    ...arch,
    posterior: normalizer > 0 ? unnormalized[idx] / normalizer : 1.0 / portfolio.length
  }));
}

/**
 * 2. Distance-Attenuated Spatial Rank Inference for Fog-of-War / Hidden Units
 * W_bar_r = sum(P(r | r_a, m) * w_a) / sum(w_a), where w_a = 1 / d_a
 */
export function computeSpatialRankPosterior(
  oldProb: number,
  neighbors: Array<{ distanceSquares: number; conditionalRankProb: number }>,
  gammaPlasticity = 0.65,
  wMax = 1.0
): number {
  if (neighbors.length === 0) return oldProb;

  let weightedSum = 0;
  let weightTotal = 0;
  let maxWeightObserved = 0;

  for (const n of neighbors) {
    const w = 1.0 / Math.max(1, n.distanceSquares);
    weightedSum += n.conditionalRankProb * w;
    weightTotal += w;
    if (w > maxWeightObserved) maxWeightObserved = w;
  }

  const wBar = weightedSum / weightTotal;
  const correctionFactor = maxWeightObserved / wMax;
  return oldProb + (wBar - oldProb) * correctionFactor * gammaPlasticity;
}

/**
 * 3. Chessformer 64-Token Geometric Attention Bias (GAB) + Source-Destination Policy Head
 * Attention_h(Q, K, V) = softmax((Q * K^T) / sqrt(d_k) + B_GAB(X)) * V
 */
export function evaluateSourceDestinationPolicy(
  squareEmbeddings: Float32Array, // 64 x d_model
  gabBiasMatrix: Float32Array,    // 64 x 64 pairwise dynamic geometry bias
  legalMoveMask: Uint8Array       // 64 x 64 binary legal action mask M_{i -> j}
): Float32Array {
  const dModel = squareEmbeddings.length / 64;
  const scale = 1.0 / Math.sqrt(dModel);
  const policyProbs = new Float32Array(64 * 64);
  let maxLogit = -Infinity;

  // Compute masked source-to-destination logits
  const logits = new Float32Array(64 * 64);
  for (let src = 0; src < 64; src++) {
    for (let dst = 0; dst < 64; dst++) {
      const idx = src * 64 + dst;
      if (legalMoveMask[idx] === 0) {
        logits[idx] = -1e9;
        continue;
      }
      let dot = 0;
      for (let k = 0; k < dModel; k++) {
        dot += squareEmbeddings[src * dModel + k] * squareEmbeddings[dst * dModel + k];
      }
      const val = dot * scale + gabBiasMatrix[idx];
      logits[idx] = val;
      if (val > maxLogit) maxLogit = val;
    }
  }

  // Softmax normalization over legal moves
  let sumExp = 0;
  for (let i = 0; i < 4096; i++) {
    if (legalMoveMask[i] === 1) {
      policyProbs[i] = Math.exp(logits[i] - maxLogit);
      sumExp += policyProbs[i];
    }
  }
  if (sumExp > 0) {
    for (let i = 0; i < 4096; i++) policyProbs[i] /= sumExp;
  }
  return policyProbs;
}

/**
 * 4. 172-Concept Linear Probe Oracle Evaluation on 1024-D Residual Stream
 */
export function probeResidualStreamConcepts(
  residual1024Master: Float32Array,
  residual1024Opponent: Float32Array,
  probeWeights: Array<{ id: string; name: string; weights: Float32Array; bias: number }>
): ConceptProbeResult[] {
  const sigmoid = (z: number) => 1.0 / (1.0 + Math.exp(-z));

  return probeWeights.map(probe => {
    let dotMaster = probe.bias;
    let dotOpp = probe.bias;
    for (let i = 0; i < 1024; i++) {
      dotMaster += residual1024Master[i] * probe.weights[i];
      dotOpp += residual1024Opponent[i] * probe.weights[i];
    }
    const masterActivation = sigmoid(dotMaster);
    const opponentActivation = sigmoid(dotOpp);
    return {
      conceptId: probe.id,
      conceptName: probe.name,
      masterActivation,
      opponentActivation,
      isExploitableBlindSpot: masterActivation > 0.85 && opponentActivation < 0.30
    };
  });
}