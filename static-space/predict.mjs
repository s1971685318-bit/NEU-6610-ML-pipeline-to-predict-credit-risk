const pay = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"];
const numbered = prefix => Array.from({ length: 6 }, (_, i) => prefix + (i + 1));
const columns = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", ...pay,
  ...numbered("BILL_AMT"), ...numbered("PAY_AMT")];
const labels = ["no_balance", "paid_duly", "revolving", "one_month_late",
  "two_months_late", "three_plus_months_late"];
const clip = (value, low, high) => Math.min(high, Math.max(low, value));
export const example = columns.join(",") + "\n" +
  "100000,2,2,1,32,0,0,0,0,0,0,20000,22000,23000,24000,25000,26000,4000,4000,4000,4000,4000,4000";

export function parseAccount(csv) {
  const rows = csv.trim().split(/\r?\n/);
  if (rows.length !== 2) throw new Error("Enter a header and one data row.");
  const header = rows[0].split(",").map(value => value.trim());
  const values = rows[1].split(",").map(value => value.trim());
  if (header.length !== values.length || new Set(header).size !== header.length) {
    throw new Error("Check the CSV headers and number of values.");
  }
  return Object.fromEntries(columns.map(name => {
    const value = values[header.indexOf(name)];
    if (!value || !/^[+-]?\d+$/.test(value) || !Number.isSafeInteger(Number(value))) {
      throw new Error("Enter an integer for " + name + ".");
    }
    return [name, Number(value)];
  }));
}

export function predict(raw, model) {
  const repayment = pay.map(name => raw[name]);
  const features = {
    ...raw,
    EDUCATION: [1, 2, 3].includes(raw.EDUCATION) ? raw.EDUCATION : 4,
    MARRIAGE: [1, 2].includes(raw.MARRIAGE) ? raw.MARRIAGE : 3,
    DELINQUENT_MONTHS: repayment.filter(value => value > 0).length,
    WORST_PAY_STATUS: Math.max(...repayment),
    CREDIT_UTILISATION: raw.LIMIT_BAL === 0 ? 0 : clip(raw.BILL_AMT1 / raw.LIMIT_BAL, -3, 5),
    PAYMENT_RATIO: clip(raw.PAY_AMT1 / Math.max(1, raw.BILL_AMT2), 0, 5),
  };
  pay.forEach((name, index) => { features[name] = labels[clip(repayment[index], -2, 3) + 2]; });
  let logit = model.intercept;
  for (const [name, [mean, scale, weight]] of Object.entries(model.numeric)) {
    logit += (features[name] - mean) / scale * weight;
  }
  for (const [name, weights] of Object.entries(model.categorical)) {
    logit += weights[features[name]] ?? 0;
  }
  const score = 1 / (1 + Math.exp(-logit));
  return { score, prediction: Number(score >= 0.5) };
}

if (typeof document !== "undefined") {
  const csv = document.querySelector("#csv");
  const result = document.querySelector("#result");
  document.querySelector("#example").onclick = () => {
    csv.value = example;
    result.textContent = "";
  };
  let model;
  document.querySelector("#predict").onclick = async () => {
    try {
      const account = parseAccount(csv.value);
      if (!model) {
        const response = await fetch("./model.json");
        if (!response.ok) throw new Error("Could not load model.json.");
        model = await response.json();
      }
      const { score, prediction } = predict(account, model);
      result.textContent = "Default score: " + score.toFixed(4) +
        " | Predicted class: " + prediction + (prediction ? " (default)" : " (no default)");
    } catch (error) {
      result.textContent = error.message;
    }
  };
}
