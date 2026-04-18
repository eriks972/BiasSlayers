"use client";

import { useState } from "react";

export default function Home() {
  const [text, setText] = useState("");
  const [mode, setMode] = useState("validity");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const analyze = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  const modes = ["validity", "bias", "tone", "combined"];

  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col items-center p-8">
      {/* Header */}
      <h1 className="text-4xl font-bold mb-2">🧠 BiasSlayers</h1>
      <p className="text-gray-400 mb-6">
        Analyze text for Validity, Bias, and Tone
      </p>

      {/* Input */}
      <textarea
        className="w-full max-w-3xl h-40 p-4 rounded-lg bg-gray-800 border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
        placeholder="Paste text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      {/* Mode Selector */}
      <div className="flex gap-3 mt-6">
        {modes.map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition ${
              mode === m
                ? "bg-blue-600"
                : "bg-gray-800 hover:bg-gray-700"
            }`}
          >
            {m.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Analyze Button */}
      <button
        onClick={analyze}
        className="mt-6 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-semibold"
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {/* Results */}
      {result && (
        <div className="mt-8 w-full max-w-3xl bg-gray-900 border border-gray-700 rounded-xl p-6 shadow-lg">
          <h2 className="text-xl font-bold mb-4">Results</h2>

          <div className="flex justify-between">
            <span className="text-gray-400">Prediction</span>
            <span className={`font-bold ${result.label === "Real" ? "text-green-400" : "text-red-400"}`}>
              {result.label}
            </span>
          </div>

          <div className="flex justify-between mt-2">
            <span className="text-gray-400">Confidence</span>
            <span>{(result.confidence * 100).toFixed(1)}%</span>
          </div>

          <div className="mt-4 w-full bg-gray-700 rounded-full h-3">
            <div
              className={`h-3 rounded-full ${result.label === "Real" ? "bg-green-500" : "bg-red-500"}`}
              style={{ width: `${result.confidence * 100}%` }}
            />
          </div>

          <br></br>

          <span className="text-gray-400 text-sm mt-1 block">
            Breakdown
          </span>

          <div className="flex justify-between mt-4">
            <span className="text-gray-400">Fake Probability</span>
            <span>{(result.fake_prob * 100).toFixed(1)}%</span>
          </div>

          <div className="flex justify-between mt-2">
            <span className="text-gray-400">Real Probability</span>
            <span>{(result.real_prob * 100).toFixed(1)}%</span>
          </div>

          <p className="mt-4 text-sm text-gray-400">
            {result.explanation}
          </p>
        </div>
      )}
    </main>
  );
}
