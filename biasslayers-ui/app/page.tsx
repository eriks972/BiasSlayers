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
     const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict_all`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
     });

    const data = await res.json();
    setResult(data);
    if (data.preview) {
      setText(data.preview);   // 👈 fill textarea automatically
    }
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

      

      <div className="space-y-3 mb-4">

        {/* URL Input */}
        <input
          type="text"
          placeholder="Paste article URL..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="w-full p-3 rounded-lg bg-slate-800 border border-slate-600 text-white"
        />

        {/* Divider */}
        <div className="text-center text-gray-400 text-sm">OR</div>

        {/* Text Area */}
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste article text here..."
          className="w-full max-w-2xl mx-auto block mt-4 p-4 
                    bg-slate-800 text-white rounded-lg 
                    border border-slate-600
                    min-h-[180px] max-h-[1000px]
                    resize-y overflow-y-auto"
          style={{ width: "1000px" }}
        />

      </div>

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
  <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-6">

  {/* BERT */}
  {result?.bert && mode !== "tone" && mode !== "bias" && (
    <div className="bg-slate-800 p-5 rounded-xl shadow-md border border-slate-700">
      <h3 className="text-lg font-semibold text-blue-400 mb-2">BERT</h3>
        <>
          <p className="text-sm text-gray-300">
        Prediction: <span className="font-medium">{result.bert.label}</span>
      </p>
          <p className="text-sm text-gray-300">
        Confidence: {(result.bert.confidence * 100).toFixed(1)}%
      </p>
        </>
    </div>
  )}

  {/* RoBERTa */}
  {result?.roberta && mode !== "tone" && mode !== "bias" && (
  <div className="bg-slate-800 p-5 rounded-xl shadow-md border border-slate-700">
    <h3 className="text-lg font-semibold text-purple-400 mb-2">RoBERTa</h3>
    <p className="text-sm text-gray-300">
      Prediction: <span className="font-medium">{result.roberta.label}</span>
    </p>
    <p className="text-sm text-gray-300">
      Confidence: {(result.roberta.confidence * 100).toFixed(1)}%
    </p>
  </div>
)}

  {/* FINAL DECISION */}
  {result?.combined && mode !== "tone" && mode !== "bias" && (
  <div className="bg-slate-900 p-6 rounded-xl shadow-lg border-2 border-green-500">
    <h3 className="text-xl font-bold text-green-400 mb-3">
      Final Decision
    </h3>

    <p className="text-lg">
      {result.combined.label === "Real" ? (
        <span className="text-green-400 font-semibold">Real</span>
      ) : (
        <span className="text-red-400 font-semibold">Fake</span>
      )}
    </p>

    <p className="text-sm text-gray-300 mt-1">
      Confidence: {(result.combined.confidence * 100).toFixed(1)}%
    </p>

    {!result.combined.agreement && (
      <p className="text-yellow-400 text-xs mt-2">
        ⚠ Models disagree — lower confidence
      </p>
    )}
  </div>
)}

  {result?.tone && mode !== "validity" && mode !== "bias" && (
  <div className="bg-slate-800 p-4 rounded-xl mt-4 text-center">
    <h3 className="text-lg font-semibold text-yellow-400">Tone</h3>

    <p className="text-white text-xl">
      {result.tone.tone}
    </p>

    <p className="text-gray-400">
      Confidence: {(result.tone.confidence * 100).toFixed(1)}%
    </p>

    <div className="text-xs text-gray-500 mt-2">
      Neg: {result.tone.negative} | Neu: {result.tone.neutral} | Pos: {result.tone.positive}
    </div>
  </div>
)}

{result.sentence_tone && mode !== "validity" && mode !== "bias" && (
  <div className="mt-6 w-full max-w-[1000px]">
    <h3 className="text-lg font-semibold text-yellow-400 mb-3">
      Tone Highlighted Text
    </h3>

    <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700 leading-relaxed">
      {result.sentence_tone.map((item: any, index: number) => (
        <span
          key={index}
          className={`mr-1 ${
            item.tone === "Positive"
              ? "text-green-400"
              : item.tone === "Negative"
              ? "text-red-400"
              : "text-gray-300"
          }`}
        >
          {item.sentence}
        </span>
      ))}
    </div>
  </div>
)}

{result.bias && mode !== "validity" && mode !== "tone" && (
  <div className="bg-slate-800 p-4 rounded-xl">
    <h3 className="text-orange-400 font-semibold mb-2">Bias</h3>

    <p className="text-lg">
      {result.bias.label}
    </p>

    <p className="text-sm text-gray-400">
      Confidence: {(result.bias.confidence * 100).toFixed(1)}%
    </p>
  </div>
)}

{result.bias_explanation && mode !== "validity" && mode !== "tone" &&  (
  <div className="bg-gray-800 p-4 rounded-xl mt-4">
    <h3 className="text-orange-400 font-semibold mb-2">Bias Explanation</h3>

    <p><strong>Summary:</strong> {result.bias_explanation.summary}</p>

    <p><strong>Tone:</strong> {result.bias_explanation.tone_skew}</p>

    <p><strong>Entities:</strong> {result.bias_explanation.entities.join(", ")}</p>

    <p><strong>Loaded Words:</strong> {result.bias_explanation.loaded_words.join(", ")}</p>
  </div>
)}

</div>
)}
    </main>
  );
}
