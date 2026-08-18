const NEUTRAL_THRESHOLD = 0.05;

const POSITIVE_WORDS = new Set([
  "good", "great", "excellent", "amazing", "awesome", "love", "loved", "loving",
  "best", "happy", "fantastic", "wonderful", "brilliant", "perfect", "nice",
  "thrilled", "delighted", "outstanding", "superb", "beautiful", "enjoy",
  "enjoyed", "recommend", "recommended", "impressive", "satisfied", "pleased",
]);

const NEGATIVE_WORDS = new Set([
  "bad", "terrible", "awful", "horrible", "worst", "hate", "hated", "poor",
  "disappointing", "disappointed", "angry", "upset", "useless", "broken",
  "waste", "wasted", "fail", "failed", "ugly", "slow", "rude", "unhappy",
  "frustrating", "frustrated", "annoying", "annoyed", "disgusting",
]);

function scoreToLabel(score) {
  if (score >= NEUTRAL_THRESHOLD) return "positive";
  if (score <= -NEUTRAL_THRESHOLD) return "negative";
  return "neutral";
}

function normalizeScores(pos, neg, neu) {
  const total = pos + neg + neu;
  if (total === 0) return { positive: 0.33, negative: 0.33, neutral: 0.34 };
  return {
    positive: round(pos / total, 4),
    negative: round(neg / total, 4),
    neutral: round(neu / total, 4),
  };
}

function round(value, digits = 4) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function analyzeVader(text, vader) {
  const scores = vader.SentimentIntensityAnalyzer.polarity_scores(text);
  const compound = scores.compound;
  const label = scoreToLabel(compound);
  const confidence = Math.min(Math.abs(compound), 1);

  return {
    model: "VADER",
    label,
    confidence: round(confidence),
    scores: {
      positive: round(scores.pos),
      negative: round(scores.neg),
      neutral: round(scores.neu),
      compound: round(compound),
    },
  };
}

function analyzeLexicon(text) {
  const words = new Set(
    text.match(/\b\w+\b/g)?.map((w) => w.replace(/[.,!?;:'"]/g, "").toLowerCase()) || []
  );

  let posHits = 0;
  let negHits = 0;
  for (const word of words) {
    if (POSITIVE_WORDS.has(word)) posHits += 1;
    if (NEGATIVE_WORDS.has(word)) negHits += 1;
  }

  const totalHits = posHits + negHits;
  let score = 0;
  let confidence = 0.35;

  if (totalHits > 0) {
    score = (posHits - negHits) / totalHits;
    confidence = Math.min(Math.abs(score) + totalHits * 0.1, 1);
  }

  const label = scoreToLabel(score);
  const pos = Math.max(score, 0);
  const neg = Math.max(-score, 0);
  const neu = Math.max(1 - pos - neg, 0);

  return {
    model: "Lexicon",
    label,
    confidence: round(confidence),
    scores: {
      ...normalizeScores(pos, neg, neu),
      positive_hits: posHits,
      negative_hits: negHits,
      score: round(score),
    },
  };
}

function ensembleVote(results) {
  const votes = { positive: 0, negative: 0, neutral: 0 };
  for (const result of results) {
    votes[result.label] += result.confidence;
  }
  const finalLabel = Object.keys(votes).reduce((a, b) => (votes[a] >= votes[b] ? a : b));
  const total = Object.values(votes).reduce((sum, v) => sum + v, 0) || 1;
  return { finalLabel, finalConfidence: votes[finalLabel] / total };
}

function computeConsistency(results) {
  if (results.length < 2) return 100;

  const labels = results.map((r) => r.label);
  const counts = {};
  for (const label of labels) counts[label] = (counts[label] || 0) + 1;
  const mostCommon = Object.keys(counts).reduce((a, b) => (counts[a] >= counts[b] ? a : b));
  const agreementCount = counts[mostCommon];
  const baseScore = (agreementCount / labels.length) * 100;
  const confidences = results.filter((r) => r.label === mostCommon).map((r) => r.confidence);
  const avgConf = confidences.reduce((sum, v) => sum + v, 0) / confidences.length || 0.5;

  return Math.min(baseScore * (0.7 + 0.3 * avgConf), 100);
}

function computeReliability(results, consistency, text) {
  const avgConfidence = results.reduce((sum, r) => sum + r.confidence, 0) / results.length;
  const wordCount = (text.match(/\b\w+\b/g) || []).length;

  let lengthFactor = 1;
  if (wordCount < 3) lengthFactor = 0.5;
  else if (wordCount < 8) lengthFactor = 0.75;

  return Math.min(
    consistency * 0.45 + avgConfidence * 100 * 0.35 + lengthFactor * 100 * 0.2,
    100
  );
}

function labelAgreement(results) {
  const counts = { positive: 0, negative: 0, neutral: 0 };
  for (const result of results) counts[result.label] += 1;
  return counts;
}

function analyzeText(text, vader) {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("Text cannot be empty");

  const models = [analyzeVader(trimmed, vader), analyzeLexicon(trimmed)];
  const { finalLabel, finalConfidence } = ensembleVote(models);
  const consistency = computeConsistency(models);
  const reliability = computeReliability(models, consistency, trimmed);

  return {
    text: trimmed,
    final_label: finalLabel,
    final_confidence: round(finalConfidence),
    consistency_score: round(consistency, 2),
    reliability_score: round(reliability, 2),
    models,
    breakdown: {
      agreement: labelAgreement(models),
      dominant_sentiment: finalLabel,
      text_length: trimmed.length,
      word_count: (trimmed.match(/\b\w+\b/g) || []).length,
      models_used: models.length,
    },
  };
}

module.exports = { analyzeText };
