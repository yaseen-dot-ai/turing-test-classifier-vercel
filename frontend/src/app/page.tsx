"use client";
import React, { useState } from "react";

const MODEL_DISPLAY_NAMES = [
  "gpt-4.1",
  "claude-3-7-sonnet-20250219",
  "roberta-large-llm-response-detector"
];

const LUX_BG =
  "bg-gradient-to-br from-[#18122B] via-[#393053] to-[#443C68] dark:from-[#18122B] dark:via-[#393053] dark:to-[#443C68]";
const CARD_BG =
  "bg-white/80 dark:bg-[#232136]/80 backdrop-blur-md shadow-xl border border-white/10 dark:border-white/10";
const FONT = "font-sans";

function validatePredictJson(input: string): { texts: string[] } | null {
  try {
    const parsed = JSON.parse(input);
    if (Array.isArray(parsed)) {
      if (parsed.every((t) => typeof t === "string")) return { texts: parsed };
    } else if (
      parsed &&
      typeof parsed === "object" &&
      Array.isArray(parsed.texts) &&
      parsed.texts.every((t: any) => typeof t === "string")
    ) {
      return { texts: parsed.texts };
    }
    return null;
  } catch {
    return null;
  }
}

function validateRunJson(input: string): any | null {
  try {
    const parsed = JSON.parse(input);
    if (
      parsed &&
      typeof parsed === "object" &&
      Array.isArray(parsed.samples) &&
      parsed.samples.every(
        (s: any) =>
          typeof s.text === "string" &&
          ["HUMAN", "AI", "AMBIGUOUS"].includes(s.label)
      )
    ) {
      return parsed;
    }
    return null;
  } catch {
    return null;
  }
}

export default function Home() {
  const [mode, setMode] = useState<"predict" | "run">("predict");

  // Predict mode state
  const [predictJson, setPredictJson] = useState(`{
  "texts": [
    "This is a human-written text.",
    "This is an AI-generated response."
  ]
}`);
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictResult, setPredictResult] = useState<any[]>([]);
  const [predictError, setPredictError] = useState<string | null>(null);

  // Run mode state
  const [runJson, setRunJson] = useState(`{
  "samples": [
    { "text": "This is a human-written text.", "label": "HUMAN" },
    { "text": "This is an AI-generated response.", "label": "AI" }
  ],
  "return_preds": true
}`);
  const [runLoading, setRunLoading] = useState(false);
  const [runResult, setRunResult] = useState<any | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  // Predict API call
  async function handlePredict() {
    setPredictLoading(true);
    setPredictError(null);
    setPredictResult([]);
    const valid = validatePredictJson(predictJson);
    if (!valid) {
      setPredictError("Invalid JSON or schema. Must be an array of strings or { texts: string[] }");
      setPredictLoading(false);
      return;
    }
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(valid),
      });
      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      setPredictResult(data.predictions || []);
    } catch (e: any) {
      setPredictError(e.message || "Unknown error");
    } finally {
      setPredictLoading(false);
    }
  }

  // Run API call
  async function handleRun() {
    setRunLoading(true);
    setRunError(null);
    setRunResult(null);
    const valid = validateRunJson(runJson);
    if (!valid) {
      setRunError(
        "Invalid JSON or schema. Must be { samples: [{ text, label }], ... } with label in HUMAN, AI, AMBIGUOUS"
      );
      setRunLoading(false);
      return;
    }
    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(valid),
      });
      if (!res.ok) throw new Error("API error");
      const data = await res.json();
      setRunResult(data);
    } catch (e: any) {
      setRunError(e.message || "Unknown error");
    } finally {
      setRunLoading(false);
    }
  }

  return (
    <div className={`${LUX_BG} ${FONT} min-h-screen py-10 px-2 flex flex-col items-center transition-colors duration-300`}>
      <div className="w-full max-w-4xl flex justify-between items-center mb-8">
        <div className="flex gap-4">
          <button
            className={`px-6 py-2 rounded-full font-semibold border transition-all ${
              mode === "predict"
                ? "bg-gradient-to-r from-[#443C68] to-[#635985] text-white border-[#443C68] shadow-lg"
                : "bg-white/80 dark:bg-[#232136]/80 text-[#443C68] border-[#443C68] hover:bg-[#443C68]/10"
            }`}
            onClick={() => setMode("predict")}
          >
            Predict
          </button>
          <button
            className={`px-6 py-2 rounded-full font-semibold border transition-all ${
              mode === "run"
                ? "bg-gradient-to-r from-[#443C68] to-[#635985] text-white border-[#443C68] shadow-lg"
                : "bg-white/80 dark:bg-[#232136]/80 text-[#443C68] border-[#443C68] hover:bg-[#443C68]/10"
            }`}
            onClick={() => setMode("run")}
          >
            Run Experiment
          </button>
        </div>
      </div>

      {mode === "predict" && (
        <div className={`w-full max-w-2xl ${CARD_BG} rounded-2xl p-8 flex flex-col gap-6 transition-all duration-300`}> 
          <h2 className="text-2xl font-bold mb-2 text-[#443C68] dark:text-white">Predict Labels</h2>
          <label className="text-sm text-[#443C68] dark:text-white font-semibold mb-1">Paste JSON (array of strings)</label>
          <textarea
            className="w-full border border-[#443C68]/30 dark:border-white/20 rounded-lg p-3 min-h-[120px] font-mono bg-white/60 dark:bg-[#232136]/60 text-[#18122B] dark:text-white shadow-inner focus:outline-none focus:ring-2 focus:ring-[#443C68] transition-all"
            value={predictJson}
            onChange={(e) => setPredictJson(e.target.value)}
            spellCheck={false}
          />
          <button
            className="bg-gradient-to-r from-[#443C68] to-[#635985] text-white px-6 py-2 rounded-lg font-semibold shadow-lg disabled:opacity-50 transition-all"
            onClick={handlePredict}
            disabled={predictLoading || !predictJson.trim()}
          >
            {predictLoading ? "Predicting..." : "Predict"}
          </button>
          {predictError && <div className="text-red-600 font-semibold">{predictError}</div>}
          {predictResult.length > 0 && (
            <div className="overflow-x-auto mt-4">
              <table className="min-w-full border text-sm rounded-xl overflow-hidden">
                <thead className="bg-[#443C68]/10 dark:bg-[#393053]/40">
                  <tr>
                    <th className="border px-2 py-1">Text</th>
                    {MODEL_DISPLAY_NAMES.map((m) => (
                      <th key={m} className="border px-2 py-1">{m}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {predictResult.map((row, i) => (
                    <tr key={i} className="hover:bg-[#443C68]/10 dark:hover:bg-[#393053]/40 transition-all">
                      <td className="border px-2 py-1 max-w-xs truncate" title={row.text}>{row.text}</td>
                      {MODEL_DISPLAY_NAMES.map((m) => (
                        <td key={m} className="border px-2 py-1">{row[m]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {mode === "run" && (
        <div className={`w-full max-w-3xl ${CARD_BG} rounded-2xl p-8 flex flex-col gap-6 transition-all duration-300`}> 
          <h2 className="text-2xl font-bold mb-2 text-[#443C68] dark:text-white">Run Experiment</h2>
          <label className="text-sm text-[#443C68] dark:text-white font-semibold mb-1">Paste JSON (see docs for schema)</label>
          <textarea
            className="w-full border border-[#443C68]/30 dark:border-white/20 rounded-lg p-3 min-h-[120px] font-mono bg-white/60 dark:bg-[#232136]/60 text-[#18122B] dark:text-white shadow-inner focus:outline-none focus:ring-2 focus:ring-[#443C68] transition-all"
            value={runJson}
            onChange={(e) => setRunJson(e.target.value)}
            spellCheck={false}
          />
          <button
            className="bg-gradient-to-r from-[#443C68] to-[#635985] text-white px-6 py-2 rounded-lg font-semibold shadow-lg disabled:opacity-50 transition-all"
            onClick={handleRun}
            disabled={runLoading || !runJson.trim()}
          >
            {runLoading ? "Running..." : "Run Experiment"}
          </button>
          {runError && <div className="text-red-600 font-semibold">{runError}</div>}
          {runResult && (
            <div className="mt-4">
              <div className="mb-2">
                <span className="font-semibold text-[#443C68] dark:text-white">Winner:</span> {runResult.winner}
              </div>
              <div className="mb-2">
                <span className="font-semibold text-[#443C68] dark:text-white">Precision:</span> {JSON.stringify(runResult.precision)}
              </div>
              <div className="mb-2">
                <span className="font-semibold text-[#443C68] dark:text-white">Recall:</span> {JSON.stringify(runResult.recall)}
              </div>
              {runResult.predictions && (
                <div className="overflow-x-auto mt-4">
                  <table className="min-w-full border text-sm rounded-xl overflow-hidden">
                    <thead className="bg-[#443C68]/10 dark:bg-[#393053]/40">
                      <tr>
                        <th className="border px-2 py-1">Text</th>
                        <th className="border px-2 py-1">Label</th>
                        {MODEL_DISPLAY_NAMES.map((m) => (
                          <th key={m} className="border px-2 py-1">{m}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {runResult.predictions.map((row: any, i: number) => (
                        <tr key={i} className="hover:bg-[#443C68]/10 dark:hover:bg-[#393053]/40 transition-all">
                          <td className="border px-2 py-1 max-w-xs truncate" title={row.text}>{row.text}</td>
                          <td className="border px-2 py-1">{row.label}</td>
                          {MODEL_DISPLAY_NAMES.map((m) => (
                            <td key={m} className="border px-2 py-1">{row[`${m}_pred`]}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
