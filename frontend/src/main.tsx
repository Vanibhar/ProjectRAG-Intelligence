import { FormEvent, KeyboardEvent, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Source = { text: string; source: string; score: number };
type Reply = { answer: string; classification: string; supported: boolean; sources: Source[] };
type Message = { question: string; reply: Reply };
const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function Sources({ sources }: { sources: Source[] }) {
  return <section className="mt-5 border-t border-slate-200 pt-4"><h3 className="text-sm font-semibold text-slate-700">Retrieved sources</h3>
    <div className="mt-3 space-y-3">{sources.map((source, index) => <article key={`${source.source}-${index}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <div className="mb-1 flex justify-between gap-3 text-xs font-semibold text-indigo-700"><span>{source.source}</span><span>Score {source.score.toFixed(2)}</span></div>
      <p className="text-sm leading-6 text-slate-600">{source.text}</p>
    </article>)}</div>
  </section>;
}

function App() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  async function submit(event?: FormEvent) {
    event?.preventDefault(); const trimmed = question.trim();
    if (!trimmed || loading) return;
    setLoading(true); setError("");
    try {
      const response = await fetch(`${API}/api/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: trimmed }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The assistant could not answer this question.");
      setHistory((items) => [...items, { question: trimmed, reply: data }]); setQuestion("");
    } catch (reason) {
      setError(reason instanceof TypeError
        ? "Cannot reach the Legal AI API. Start it with: python -m uvicorn app:app --port 8000"
        : reason instanceof Error ? reason.message : "Network error. Check that the API is running.");
    }
    finally { setLoading(false); }
  }
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(); } }
  return <main className="min-h-screen bg-slate-100 text-slate-900"><div className="mx-auto flex min-h-screen max-w-4xl flex-col px-4 py-8 sm:px-6">
    <header className="mb-7"><p className="text-sm font-bold uppercase tracking-[0.2em] text-indigo-700">Constitutional research</p><h1 className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">Legal AI Assistant</h1><p className="mt-2 text-slate-600">Ask questions against the indexed Constitution corpus.</p></header>
    <section className="flex-1 space-y-5">{history.length === 0 && !loading && <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center text-slate-500">Ask a legal question to begin.</div>}
      {history.map((item, index) => <div key={index} className="space-y-3"><div className="ml-auto max-w-2xl rounded-2xl rounded-tr-sm bg-indigo-700 px-4 py-3 text-white">{item.question}</div><article className="max-w-3xl rounded-2xl rounded-tl-sm bg-white p-5 shadow-sm"><div className="mb-3 text-xs font-semibold uppercase tracking-wide text-indigo-700">{item.reply.classification.replaceAll("_", " ")}</div><p className={`whitespace-pre-wrap leading-7 ${item.reply.supported ? "text-slate-700" : "text-slate-600 italic"}`}>{item.reply.answer}</p>{item.reply.supported && item.reply.sources.length > 0 && <Sources sources={item.reply.sources} />}</article></div>)}
      {loading && <div className="flex items-center gap-3 rounded-xl bg-white p-4 text-slate-600 shadow-sm"><span className="h-4 w-4 animate-spin rounded-full border-2 border-indigo-200 border-t-indigo-700" />Retrieving legal sources and drafting an answer…</div>}
    </section>
    {error && <p role="alert" className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p>}
    <form onSubmit={submit} className="sticky bottom-0 mt-6 rounded-xl border border-slate-200 bg-white p-3 shadow-lg"><label htmlFor="question" className="sr-only">Your legal question</label><textarea id="question" value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={onKeyDown} rows={3} placeholder="For example: What is Article 21?" className="w-full resize-none rounded-lg border border-slate-200 p-3 outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100" />
      <div className="mt-2 flex items-center justify-between gap-3"><span className="text-xs text-slate-500">Enter to ask · Shift+Enter for a new line</span><div className="flex gap-2"><button type="button" onClick={() => { setQuestion(""); setError(""); }} className="rounded-lg px-3 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100">Clear</button><button disabled={!question.trim() || loading} className="rounded-lg bg-indigo-700 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-800 disabled:cursor-not-allowed disabled:opacity-50">Ask</button></div></div>
    </form></div></main>;
}
createRoot(document.getElementById("root")!).render(<App />);
