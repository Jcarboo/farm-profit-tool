'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import Link from "next/link";
import {
  LineChart, Line, ResponsiveContainer, CartesianGrid, XAxis, YAxis, Tooltip,
  BarChart, Bar,
  Label,
  Legend
} from 'recharts';

type FeatureKey = 'Seed'|'Fertilizer'|'Chemicals'|'Services'|'FLE'|'Repairs'|'Water'|'Interest';
const FEATURES: FeatureKey[] = ['Seed','Fertilizer','Chemicals','Services','FLE','Repairs','Water','Interest'];

type InputsNum = Record<FeatureKey, number>;
type InputsStr = Record<FeatureKey, string>;
type CropInputsNum = Record<string, InputsNum>;
type CropInputsStr = Record<string, InputsStr>;

type PredictResponse = {
  crop: string;
  prediction: number;
  interval_80: [number|null, number|null];
  interval_95: [number|null, number|null];
  contributions: Record<FeatureKey, number>;
  marginal_per_dollar: Record<FeatureKey, number>;
  model_meta: any;
};

type PredictMultiResponse = {
  aggregate_prediction: number;
  aggregate_contributions: Record<FeatureKey, number>;
  per_crop: Array<{
    crop: string;
    share: number;
    prediction: number;
    interval_80: [number|null, number|null];
    interval_95: [number|null, number|null];
  }>;
};

const toCurrency = (n: number | null | undefined) =>
  (n === null || n === undefined || Number.isNaN(n)) ? '—' :
  n.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });

const greenPanel = "bg-emerald-900/30 border border-emerald-700/60 rounded-2xl";

// ---------- helpers ----------
const zeroNum = (): InputsNum => ({ Seed:0, Fertilizer:0, Chemicals:0, Services:0, FLE:0, Repairs:0, Water:0, Interest:0 });
const emptyStrs = (): InputsStr => ({ Seed:'', Fertilizer:'', Chemicals:'', Services:'', FLE:'', Repairs:'', Water:'', Interest:'' });

const toNum = (v: string | number | undefined): number => {
  if (typeof v === 'number') return Number.isFinite(v) ? v : 0;
  const s = (v ?? '').toString().trim();
  if (s === '' || s === '-' || s === '.' || s === '-.') return 0;
  const n = parseFloat(s.replace(/,/g,''));
  return Number.isFinite(n) ? n : 0;
};

export default function Page() {
  const [crops, setCrops] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);

  // display inputs (strings so user can clear / type freely)
  const [inputsStr, setInputsStr] = useState<CropInputsStr>({});
  // what-if deltas (numbers, used by sliders)
  const [whatIf, setWhatIf] = useState<CropInputsNum>({});
  // shares (string to avoid stuck 0)
  const [shareStr, setShareStr] = useState<Record<string, string>>({});

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string>('');

  const [singleResult, setSingleResult] = useState<PredictResponse|null>(null);
  const [multiResult, setMultiResult] = useState<PredictMultiResponse|null>(null);

  // what-if live results
  const [whatIfSingle, setWhatIfSingle] = useState<PredictResponse|null>(null);
  const [whatIfMulti, setWhatIfMulti] = useState<PredictMultiResponse|null>(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  // sparkline history
  const [history, setHistory] = useState<number[]>([]);

  // load crops
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch('/api/crops');
        const arr: string[] = await r.json();
        setCrops(arr);
        if (arr.length && selected.length === 0) {
          const first = arr[0];
          setSelected([first]);
          setInputsStr({ [first]: emptyStrs() });
          setWhatIf({ [first]: zeroNum() });
          setShareStr({ [first]: '1' });
        }
      } catch (e: any) {
        setErr(e?.message || 'Failed to load crops');
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const multiMode = selected.length > 1;
  const sharesNum = useMemo(() => {
    const out: Record<string, number> = {};
    for (const c of selected) out[c] = toNum(shareStr[c]);
    return out;
  }, [selected, shareStr]);

  const totalShare = useMemo(() => {
    return selected.reduce((acc, c) => acc + (sharesNum[c] || 0), 0);
  }, [selected, sharesNum]);

  const ensureCropInit = (crop: string) => {
    setInputsStr(prev => prev[crop] ? prev : { ...prev, [crop]: emptyStrs() });
    setWhatIf(prev => prev[crop] ? prev : { ...prev, [crop]: zeroNum() });
    setShareStr(prev => (crop in prev) ? prev : { ...prev, [crop]: '1' });
  };

  const toggleCrop = (crop: string) => {
    setErr('');
    setSingleResult(null); setMultiResult(null);
    setWhatIfSingle(null); setWhatIfMulti(null);
    setHistory([]);
    setSelected(prev => {
      if (prev.includes(crop)) {
        const next = prev.filter(c => c !== crop);
        return next.length ? next : [];
      } else {
        ensureCropInit(crop);
        return [...prev, crop];
      }
    });
  };

  const updateInputStr = (crop: string, key: FeatureKey, v: string) => {
    setInputsStr(prev => ({
      ...prev,
      [crop]: { ...(prev[crop] || emptyStrs()), [key]: v }
    }));
  };

  const updateShareStr = (crop: string, v: string) => {
    setShareStr(prev => ({ ...prev, [crop]: v }));
  };

  const updateWhatIf = (crop: string, key: FeatureKey, v: string) => {
    const num = toNum(v);
    setWhatIf(prev => ({
      ...prev,
      [crop]: { ...(prev[crop] || zeroNum()), [key]: num }
    }));
  };

  const resetWhatIf = () => {
    setWhatIf(prev => {
      const next: CropInputsNum = {};
      for (const c of selected) next[c] = zeroNum();
      return next;
    });
    setWhatIfSingle(null); setWhatIfMulti(null);
    if (singleResult) setHistory([singleResult.prediction]);
    if (multiResult) setHistory([multiResult.aggregate_prediction]);
  };

  const inputsNumForAPI = (crop: string): InputsNum => {
    const baseStr = inputsStr[crop] || emptyStrs();
    const out: any = {};
    for (const f of FEATURES) out[f] = toNum(baseStr[f]);
    return out as InputsNum;
  };

  const inputsWithWhatIf = (crop: string): InputsNum => {
    const base = inputsNumForAPI(crop);
    const delta = whatIf[crop] || zeroNum();
    const merged: any = {};
    for (const f of FEATURES) merged[f] = (base[f] ?? 0) + (delta[f] ?? 0);
    return merged as InputsNum;
  };

  const estimate = async () => {
    try {
      setLoading(true);
      setErr('');
      setSingleResult(null); setMultiResult(null);
      setWhatIfSingle(null); setWhatIfMulti(null);
      setHistory([]);

      if (!selected.length) {
        setErr('Pick at least one crop.');
        return;
      }

      if (!multiMode) {
        const crop = selected[0];
        const body = { crop, inputs: inputsNumForAPI(crop) };
        const r = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!r.ok) throw new Error(await r.text());
        const data: PredictResponse = await r.json();
        setSingleResult(data);
        setHistory([data.prediction]);
      } else {
        const body = {
          crops: selected.map(crop => ({
            crop,
            share: sharesNum[crop] || 0,
            inputs: inputsNumForAPI(crop)
          }))
        };
        const r = await fetch('/api/predict-multi', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!r.ok) throw new Error(await r.text());
        const data: PredictMultiResponse = await r.json();
        setMultiResult(data);
        setHistory([data.aggregate_prediction]);
      }
    } catch (e: any) {
      setErr(e?.message || 'Failed to predict');
    } finally {
      setLoading(false);
    }
  };

  // Debounced What-If compute + append to sparkline
  useEffect(() => {
    const hasBaseline = (!multiMode && !!singleResult) || (multiMode && !!multiResult);
    if (!hasBaseline || selected.length === 0) return;

    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      try {
        if (!multiMode) {
          const crop = selected[0];
          const body = { crop, inputs: inputsWithWhatIf(crop) };
          const r = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
          if (!r.ok) throw new Error(await r.text());
          const data: PredictResponse = await r.json();
          setWhatIfSingle(data);
          setHistory(prev => [...prev.slice(-19), data.prediction]); // keep last 20
        } else {
          const body = {
            crops: selected.map(crop => ({
              crop,
              share: sharesNum[crop] || 0,
              inputs: inputsWithWhatIf(crop)
            }))
          };
          const r = await fetch('/api/predict-multi', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
          if (!r.ok) throw new Error(await r.text());
          const data: PredictMultiResponse = await r.json();
          setWhatIfMulti(data);
          setHistory(prev => [...prev.slice(-19), data.aggregate_prediction]);
        }
      } catch {
        // ignore in UI
      }
    }, 400);

    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [whatIf, selected, sharesNum, inputsStr, multiMode, singleResult, multiResult]);

  const recFromMarginals = (m: Record<string, number> | undefined) => {
    if (!m) return { ups: [], downs: [] };
    const pairs = Object.entries(m) as Array<[FeatureKey, number]>;
    const ups = pairs.filter(([,v]) => v > 0).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([k,v]) => `${k} (+${v.toFixed(2)}/$)`);
    const downs = pairs.filter(([,v]) => v < 0).sort((a,b)=>a[1]-b[1]).slice(0,3).map(([k,v]) => `${k} (${v.toFixed(2)}/$)`);
    return { ups, downs };
  };

  // ---------- UI ----------
  return (
    <div className="min-h-screen bg-gradient-to-b from-emerald-950 to-green-900 text-green-50">
      <div className="max-w-6xl mx-auto px-5 py-10 space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight">
            Farm Profit Estimator
          </h1>
          <p className="text-green-200/90">
            Select crops, enter costs, and estimate year-end net returns. Cool greens, farm vibes 🌱
          </p>
        </header>

        {/* About teaser */}
          <section className={`${greenPanel} p-4`}>
            <div className="flex flex-col md:flex-row items-center gap-4">
              <img
                src="https://i.pinimg.com/originals/a5/8d/69/a58d69c83ef14e19a221d15e3510a237.gif"
                alt="Animated corn field GIF"
                className="w-28 h-28 md:w-32 md:h-32 rounded-xl object-cover border border-emerald-700 shadow-lg"
                loading="lazy"
              />
              <div className="flex-1 text-center md:text-left">
                <h2 className="text-lg md:text-xl font-semibold">Want the full story?</h2>
                <p className="text-green-200/85 mt-1">
                  Learn how I cleaned USDA data, trained crop-specific models, built
                  uncertainty bands, and turned insights into actionable recommendations.
                </p>
                <div className="mt-3">
                  <Link
                    href="/about"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-green-950 font-semibold"
                  >
                    More about the project<span aria-hidden>→</span>
                  </Link>
                </div>
              </div>
            </div>
          </section>


        {/* Crop selector */}
        <section className={`${greenPanel} p-4`}>
          <h2 className="text-xl font-semibold mb-3">1) Pick your crop(s)</h2>
          {crops.length === 0 ? (
            <div className="text-green-200/80">Loading crops…</div>
          ) : (
            <div className="flex flex-wrap gap-3">
              {crops.map(crop => {
                const active = selected.includes(crop);
                return (
                  <button
                    key={crop}
                    onClick={() => toggleCrop(crop)}
                    className={`px-4 py-2 rounded-xl border transition capitalize
                      ${active ? 'bg-emerald-600/80 border-emerald-400' : 'bg-transparent border-emerald-700 hover:bg-emerald-800/40'}`}
                  >
                    {crop}
                  </button>
                );
              })}
            </div>
          )}
          {selected.length > 1 && (
            <div className="mt-2 text-sm text-emerald-200/80">
              Multi-crop mode: shares will be normalized automatically.
            </div>
          )}
        </section>

        {/* Inputs */}
        {selected.length > 0 && (
          <section className={`${greenPanel} p-4 space-y-6`}>
            <h2 className="text-xl font-semibold">2) Enter costs</h2>

            <div className="grid gap-6 md:grid-cols-2">
              {selected.map(crop => (
                <div key={crop} className="rounded-2xl bg-green-950/40 border border-emerald-800 p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="font-semibold text-lg capitalize">{crop}</h3>
                    {selected.length > 1 && (
                      <div className="text-sm">
                        <label className="mr-2 text-green-200/80">Share</label>
                        <input
                          type="text"
                          inputMode="decimal"
                          value={shareStr[crop] ?? ''}
                          onChange={e => updateShareStr(crop, e.target.value)}
                          placeholder="1"
                          className="w-24 bg-emerald-950 border border-emerald-700 rounded-lg px-2 py-1"
                        />
                      </div>
                    )}
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {FEATURES.map(f => (
                      <div key={`${crop}-${f}`} className="flex flex-col">
                        <label className="text-xs text-green-200/80 mb-1">{f} ($)</label>
                        <input
                          type="text"
                          inputMode="decimal"
                          value={(inputsStr[crop]?.[f] ?? '')}
                          onChange={(e) => updateInputStr(crop, f, e.target.value)}
                          placeholder="0"
                          className="bg-emerald-950 border border-emerald-700 rounded-lg px-3 py-2"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {selected.length > 1 && (
              <div className="text-sm text-emerald-200/80">
                Total share (before normalization): <span className="font-semibold">
                  {selected.reduce((acc, c) => acc + (sharesNum[c] || 0), 0)}
                </span>
              </div>
            )}

            <button
              onClick={estimate}
              disabled={loading}
              className="px-5 py-3 rounded-2xl bg-emerald-500 hover:bg-emerald-400 text-green-950 font-semibold disabled:opacity-60"
            >
              {loading ? 'Estimating…' : 'Estimate profit'}
            </button>

            {err && <div className="text-red-300">{err}</div>}
          </section>
        )}

        {/* Results - Single */}
        {singleResult && (
          <section className={`${greenPanel} p-4 space-y-4`}>
            <h2 className="text-xl font-semibold">3) Results</h2>

            <div className="grid gap-4 md:grid-cols-3">
              <div className="rounded-xl bg-emerald-950/50 p-4 border border-emerald-800">
                <div className="text-emerald-300 text-sm">Predicted profit</div>
                <div className="text-3xl font-extrabold">{toCurrency(singleResult.prediction)}</div>
                <div className="text-xs text-green-200/80 mt-1">
                  80%: {toCurrency(singleResult.interval_80?.[0])} – {toCurrency(singleResult.interval_80?.[1])}<br/>
                  95%: {toCurrency(singleResult.interval_95?.[0])} – {toCurrency(singleResult.interval_95?.[1])}
                </div>
              </div>

              {/* Recharts: contributions bar */}
              <div className="rounded-xl bg-emerald-950/50 p-4 border border-emerald-800 md:col-span-2">
                <div className="text-emerald-300 text-sm mb-2">Contribution breakdown</div>
                <div className="w-full h-64 md:h-80 xl:h-[28rem]"> {/* bigger wrapper */}
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={FEATURES.map(f => ({ name: f, val: singleResult.contributions[f] }))}
                      margin={{ top: 8, right: 16, left: 16, bottom: 48 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" tick={{ fill: '#A7F3D0', fontSize: 12 }}>
                        <Label value="Cost category" position="insideBottom" offset={-28} style={{ fill: '#A7F3D0' }}/>
                      </XAxis>
                      <YAxis tickFormatter={(v) => toCurrency(v as number)} tick={{ fill: '#A7F3D0', fontSize: 12 }}>
                        {/* put the y-label outside so it doesn't overlap */}
                        <Label value="Contribution to predicted profit (USD)" angle={-90} position="left" style={{ fill: '#A7F3D0' }} />
                      </YAxis>
                      <Tooltip formatter={(v) => toCurrency(v as number)} labelFormatter={(label) => `Category: ${label}`} />
                      <Legend />
                      <Bar name="Contribution ($)" dataKey="val" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>


              </div>
            </div>

            {/* WHAT-IF (single) */}
            <section className="rounded-2xl bg-emerald-950/40 border border-emerald-800 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">4) What-If Analysis</h3>
                <button
                  onClick={resetWhatIf}
                  className="text-xs px-3 py-1 rounded-lg bg-emerald-800 hover:bg-emerald-700 border border-emerald-600"
                >
                  Reset adjustments
                </button>
              </div>
              <p className="text-sm text-emerald-200/80">
                Move sliders to adjust spend; re-predict is live.
              </p>

              {selected.length === 1 && (
                <div className="space-y-3">
                  {FEATURES.map(f => (
                    <div key={`wf-${f}`} className="grid grid-cols-1 md:grid-cols-6 gap-2 items-center">
                      <label className="md:col-span-1 text-sm">{f}</label>
                      <input
                        type="range"
                        min={-5000} max={5000} step={50}
                        value={(whatIf[selected[0]]?.[f] ?? 0)}
                        onChange={(e) => updateWhatIf(selected[0], f, e.target.value)}
                        className="md:col-span-4 accent-emerald-400"
                        title="Adjust dollars"
                      />
                      <div className="md:col-span-1 text-right text-sm tabular-nums">
                        {toCurrency((whatIf[selected[0]]?.[f] ?? 0))}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Sparkline */}
              <div className="rounded-xl bg-emerald-950/60 p-3 border border-emerald-800">
                <div className="text-sm text-emerald-300 mb-2">Prediction trend (baseline + what-if scrubs)</div>
                <div className="w-full h-64 md:h-80 xl:h-[26rem]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={history.map((v, i) => ({ step: i, value: v }))}
                      margin={{ top: 8, right: 16, left: 16, bottom: 48 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="step" tick={{ fill: '#A7F3D0', fontSize: 12 }}>
                        <Label value="What-if adjustments over time" position="insideBottom" offset={-28} style={{ fill: '#A7F3D0' }} />
                      </XAxis>
                      <YAxis tickFormatter={(v) => toCurrency(v as number)} tick={{ fill: '#A7F3D0', fontSize: 12 }}>
                        <Label value="Predicted profit (USD)" angle={-90} position="left" style={{ fill: '#A7F3D0' }} />
                      </YAxis>
                      <Tooltip formatter={(v) => toCurrency(v as number)} labelFormatter={(l) => `Step ${l}`} />
                      <Legend />
                      <Line name="Predicted profit" type="monotone" dataKey="value" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

              </div>

              {/* Adjusted vs baseline cards */}
              <div className="grid gap-4 md:grid-cols-3 pt-1">
                <div className="rounded-xl bg-emerald-950/60 p-4 border border-emerald-800">
                  <div className="text-emerald-300 text-sm">Adjusted prediction {whatIfLoading && '…'}</div>
                  <div className="text-3xl font-extrabold">
                    {toCurrency(whatIfSingle?.prediction ?? singleResult.prediction)}
                  </div>
                  <div className="text-xs text-emerald-200/80 mt-1">
                    Δ vs baseline:&nbsp;
                    <span className={(whatIfSingle && whatIfSingle.prediction - singleResult.prediction >= 0) ? 'text-emerald-300' : 'text-red-300'}>
                      {toCurrency((whatIfSingle?.prediction ?? singleResult.prediction) - singleResult.prediction)}
                    </span>
                  </div>
                </div>
                <div className="rounded-xl bg-emerald-950/60 p-4 border border-emerald-800 md:col-span-2">
                  <div className="text-sm text-emerald-300 mb-1">Adjusted ranges</div>
                  <div className="text-xs text-emerald-200/80">
                    80%: {toCurrency(whatIfSingle?.interval_80?.[0] ?? singleResult.interval_80?.[0])}
                    &nbsp;–&nbsp;
                    {toCurrency(whatIfSingle?.interval_80?.[1] ?? singleResult.interval_80?.[1])}
                    <br/>
                    95%: {toCurrency(whatIfSingle?.interval_95?.[0] ?? singleResult.interval_95?.[0])}
                    &nbsp;–&nbsp;
                    {toCurrency(whatIfSingle?.interval_95?.[1] ?? singleResult.interval_95?.[1])}
                  </div>
                </div>
              </div>
            </section>
          </section>
        )}

        {/* Results - Multi */}
        {multiResult && (
          <section className={`${greenPanel} p-4 space-y-4`}>
            <h2 className="text-xl font-semibold">3) Results (multi-crop)</h2>

            <div className="rounded-xl bg-emerald-950/50 p-4 border border-emerald-800">
              <div className="text-emerald-300 text-sm">Aggregate predicted profit</div>
              <div className="text-3xl font-extrabold">{toCurrency(multiResult.aggregate_prediction)}</div>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
              {multiResult.per_crop.map(pc => (
                <div key={pc.crop} className="rounded-xl bg-emerald-950/40 p-4 border border-emerald-800">
                  <div className="flex items-center justify-between mb-1">
                    <div className="capitalize font-semibold">{pc.crop}</div>
                    <div className="text-xs text-green-200/70">share {(pc.share*100).toFixed(1)}%</div>
                  </div>
                  <div className="text-2xl font-bold">{toCurrency(pc.prediction)}</div>
                  <div className="text-xs text-green-200/80 mt-1">
                    80%: {toCurrency(pc.interval_80?.[0])} – {toCurrency(pc.interval_80?.[1])}<br/>
                    95%: {toCurrency(pc.interval_95?.[0])} – {toCurrency(pc.interval_95?.[1])}
                  </div>
                </div>
              ))}
            </div>

            {/* What-If (multi) */}
            <section className="rounded-2xl bg-emerald-950/40 border border-emerald-800 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">4) What-If Analysis (multi-crop)</h3>
                <button
                  onClick={resetWhatIf}
                  className="text-xs px-3 py-1 rounded-lg bg-emerald-800 hover:bg-emerald-700 border border-emerald-600"
                >
                  Reset adjustments
                </button>
              </div>

              <div className="space-y-4">
                {selected.map(crop => (
                  <div key={`wf-card-${crop}`} className="rounded-xl bg-emerald-950/50 p-3 border border-emerald-800">
                    <div className="font-semibold capitalize mb-2">{crop}</div>
                    <div className="grid gap-2">
                      {FEATURES.map(f => (
                        <div key={`wf-${crop}-${f}`} className="grid grid-cols-1 md:grid-cols-6 gap-2 items-center">
                          <label className="md:col-span-1 text-sm">{f}</label>
                          <input
                            type="range"
                            min={-5000} max={5000} step={50}
                            value={(whatIf[crop]?.[f] ?? 0)}
                            onChange={(e) => updateWhatIf(crop, f, e.target.value)}
                            className="md:col-span-4 accent-emerald-400"
                          />
                          <div className="md:col-span-1 text-right text-sm tabular-nums">
                            {toCurrency((whatIf[crop]?.[f] ?? 0))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              {/* Sparkline aggregate */}
              <div className="rounded-xl bg-emerald-950/60 p-3 border border-emerald-800">
                <div className="text-sm text-emerald-300 mb-2">Aggregate prediction trend</div>
                <div className="w-full h-64 md:h-80 xl:h-[26rem]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={history.map((v, i) => ({ step: i, value: v }))}
                      margin={{ top: 8, right: 16, left: 16, bottom: 48 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="step" tick={{ fill: '#A7F3D0', fontSize: 12 }}>
                        <Label value="What-if adjustments over time" position="insideBottom" offset={-28} style={{ fill: '#A7F3D0' }} />
                      </XAxis>
                      <YAxis tickFormatter={(v) => toCurrency(v as number)} tick={{ fill: '#A7F3D0', fontSize: 12 }}>
                        <Label value="Aggregate predicted profit (USD)" angle={-90} position="left" style={{ fill: '#A7F3D0' }} />
                      </YAxis>
                      <Tooltip formatter={(v) => toCurrency(v as number)} labelFormatter={(l) => `Step ${l}`} />
                      <Legend />
                      <Line name="Aggregate profit" type="monotone" dataKey="value" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-3 pt-1">
                <div className="rounded-xl bg-emerald-950/60 p-4 border border-emerald-800">
                  <div className="text-emerald-300 text-sm">Adjusted aggregate {whatIfLoading && '…'}</div>
                  <div className="text-3xl font-extrabold">
                    {toCurrency(whatIfMulti?.aggregate_prediction ?? multiResult.aggregate_prediction)}
                  </div>
                  <div className="text-xs text-emerald-200/80 mt-1">
                    Δ vs baseline:&nbsp;
                    <span className={(whatIfMulti && (whatIfMulti.aggregate_prediction - multiResult.aggregate_prediction) >= 0) ? 'text-emerald-300' : 'text-red-300'}>
                      {toCurrency((whatIfMulti?.aggregate_prediction ?? multiResult.aggregate_prediction) - multiResult.aggregate_prediction)}
                    </span>
                  </div>
                </div>
                <div className="rounded-xl bg-emerald-950/60 p-4 border border-emerald-800 md:col-span-2">
                  <div className="text-sm text-emerald-300 mb-1">Notes</div>
                  <div className="text-xs text-emerald-200/80">
                    Shares are normalized on the server. Consider agronomic minimums for seed and essential inputs.
                    Interest and FLE reductions raise profit only if they don’t reduce production capacity.
                  </div>
                </div>
              </div>
            </section>

            <details className="rounded-xl bg-emerald-950/30 p-4 border border-emerald-800">
              <summary className="cursor-pointer text-emerald-300">Aggregate contributions</summary>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-3 text-sm">
                {FEATURES.map(f => (
                  <div key={`agg-${f}`} className="flex justify-between bg-emerald-950/50 px-3 py-2 rounded-lg">
                    <span>{f}</span>
                    <span>{(multiResult.aggregate_contributions[f] ?? 0).toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </details>
          </section>
        )}

        <footer className="text-xs text-green-200/60 pb-10">
          Models are trained per crop on historical USDA data; estimates include uncertainty and should be used as guidance, not guarantees.
        </footer>
      </div>
    </div>
  );
}
