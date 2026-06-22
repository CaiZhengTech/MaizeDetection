import { useCallback, useEffect, useRef, useState } from "react";

// ── Types ────────────────────────────────────────────────────────────────────

interface ClassProbability {
  name: string;
  key: string;
  probability: number;
}

interface Prediction {
  label: string;
  key: string;
  confidence: number;
  probabilities: ClassProbability[];
  risk: "Low" | "Moderate" | "High";
  needsReview: boolean;
}

// ── Constants ────────────────────────────────────────────────────────────────

const CLASSES = [
  {
    name: "Healthy",
    key: "healthy",
    description:
      "Vibrant green foliage with no visible lesions, discoloration, or pest damage. This is the baseline state used to calibrate the model.",
    why: "Knowing what healthy looks like is the foundation of any early-warning system. It lets farmers distinguish normal crop stress from treatable disease before yield is affected.",
    wiki: "https://en.wikipedia.org/wiki/Maize",
    symptoms: [
      "Uniform deep-green color",
      "Intact, unbroken leaf surface",
      "No pustules, lesions, or wilting",
    ],
    gradient:
      "linear-gradient(135deg, oklch(0.55 0.18 148), oklch(0.42 0.12 155))",
    placeholderColor: "oklch(0.55 0.13 148)",
  },
  {
    name: "Common Rust",
    key: "common_rust",
    description:
      "A fungal infection caused by Puccinia sorghi that produces small, cinnamon-brown pustules on both leaf surfaces. It thrives in cool, humid conditions.",
    why: "Severe infections can reduce grain yield by 10–40%. Because pustules rupture the leaf cuticle, the plant loses photosynthetic area and becomes more susceptible to secondary pathogens.",
    wiki: "https://en.wikipedia.org/wiki/Puccinia_sorghi",
    symptoms: [
      "Brick-red to cinnamon pustules",
      "Pustules on both leaf surfaces",
      "Worse in cool, humid conditions",
    ],
    gradient:
      "linear-gradient(135deg, oklch(0.52 0.16 38), oklch(0.42 0.14 28))",
    placeholderColor: "oklch(0.52 0.16 38)",
  },
  {
    name: "Northern Corn Leaf Blight",
    key: "northern_leaf_blight",
    description:
      "Caused by the fungus Exserohilum turcicum, it appears as long, cigar-shaped gray-green to tan lesions running parallel to leaf veins.",
    why: "It is one of the most economically significant foliar diseases of corn globally. Under favorable conditions, yield losses can exceed 50% in susceptible hybrids.",
    wiki: "https://en.wikipedia.org/wiki/Northern_corn_leaf_blight",
    symptoms: [
      "Long, cigar-shaped lesions",
      "Gray-green fading to tan",
      "Runs parallel to leaf veins",
    ],
    gradient:
      "linear-gradient(135deg, oklch(0.62 0.09 80), oklch(0.50 0.07 85))",
    placeholderColor: "oklch(0.62 0.09 80)",
  },
  {
    name: "Gray Leaf Spot",
    key: "gray_leaf_spot",
    description:
      "A disease caused by Cercospora zeae-maydis characterized by rectangular, gray to tan lesions with distinct parallel edges, limited by leaf veins.",
    why: "Gray leaf spot is the number one yield-limiting foliar disease in the U.S. Corn Belt. It spreads rapidly in warm, humid weather and can cause devastating yield losses if not managed early.",
    wiki: "https://en.wikipedia.org/wiki/Gray_leaf_spot",
    symptoms: [
      "Rectangular gray-tan lesions",
      "Sharp, vein-limited edges",
      "Spreads fast in warm humidity",
    ],
    gradient:
      "linear-gradient(135deg, oklch(0.52 0.04 250), oklch(0.42 0.03 260))",
    placeholderColor: "oklch(0.52 0.04 250)",
  },
];

const SAMPLE_IMAGES = [
  { id: "h1", label: "Healthy", classKey: "healthy", tag: "controlled" as const, src: "/samples/healthy_controlled.jpg" },
  { id: "cr1", label: "Common Rust", classKey: "common_rust", tag: "controlled" as const, src: "/samples/common_rust_controlled.jpg" },
  { id: "nlb1", label: "NLB", classKey: "northern_leaf_blight", tag: "controlled" as const, src: "/samples/nlb_controlled.jpg" },
  { id: "gls1", label: "Gray Leaf Spot", classKey: "gray_leaf_spot", tag: "controlled" as const, src: "/samples/gls_controlled.jpg" },
  { id: "nlb2", label: "NLB", classKey: "northern_leaf_blight", tag: "field" as const, src: "/samples/nlb_field.jpg" },
  { id: "gls2", label: "Gray Leaf Spot", classKey: "gray_leaf_spot", tag: "field" as const, src: "/samples/gls_field.jpg" },
  { id: "nlb3", label: "NLB (tricky)", classKey: "northern_leaf_blight", tag: "field" as const, src: "/samples/healthy_field.jpg" },
  { id: "gls3", label: "GLS (tricky)", classKey: "gray_leaf_spot", tag: "field" as const, src: "/samples/common_rust_field.jpg" },
];

const DOMAIN_GAP = {
  inDomain: 0.984,
  field: 0.617,
  perClass: [
    { name: "Healthy", controlled: 0.99, field: null as number | null },
    { name: "Common Rust", controlled: 0.98, field: null as number | null },
    { name: "Northern Leaf Blight", controlled: 0.98, field: 0.7 },
    { name: "Gray Leaf Spot", controlled: 0.99, field: 0.53 },
  ],
};

let draggedSample: (typeof SAMPLE_IMAGES)[number] | null = null;

// ── Inline components ────────────────────────────────────────────────────────

function LeafMark() {
  return (
    <span className="inline-flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-leaf to-corn text-primary-foreground shadow-sm">
      <svg
        viewBox="0 0 24 24"
        fill="none"
        className="h-5 w-5"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M5 19c8 0 14-6 14-14-8 0-14 6-14 14Z" />
        <path d="M5 19 19 5" />
      </svg>
    </span>
  );
}

function LeafIcon({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M5 19c8 0 14-6 14-14-8 0-14 6-14 14Z" />
      <path d="M5 19 19 5" />
    </svg>
  );
}

function Spinner({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg
      className={`animate-spin ${className}`}
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeOpacity="0.25"
        strokeWidth="3"
      />
      <path
        d="M22 12a10 10 0 0 1-10 10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

function Badge({
  tone,
  children,
}: {
  tone: "success" | "warning" | "destructive";
  children: React.ReactNode;
}) {
  const cls = {
    success: "bg-success/15 text-success border-success/30",
    warning: "bg-warning/25 text-warning-foreground border-warning/40",
    destructive: "bg-destructive/10 text-destructive border-destructive/30",
  }[tone];

  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${cls}`}
    >
      {children}
    </span>
  );
}

// ── Class info dialog ────────────────────────────────────────────────────────

function ClassInfoDialog({
  cls,
  onClose,
}: {
  cls: (typeof CLASSES)[number];
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-foreground/40 p-4"
      onClick={onClose}
    >
      <div
        className="animate-dialog-in relative w-full max-w-sm overflow-hidden rounded-2xl border border-border bg-card shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Gradient header */}
        <div
          className="relative flex items-center gap-3 px-6 py-5"
          style={{ background: cls.gradient }}
        >
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-white/90 shadow-sm">
            <LeafIcon className="h-5 w-5 text-foreground" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white">{cls.name}</h3>
            <p className="font-mono text-[11px] text-white/70">{cls.key}</p>
          </div>
          <button
            onClick={onClose}
            className="absolute right-3 top-3 flex h-7 w-7 items-center justify-center rounded-full bg-foreground/20 text-white backdrop-blur-sm hover:bg-foreground/30"
            aria-label="Close"
          >
            <svg
              viewBox="0 0 24 24"
              className="h-3.5 w-3.5"
              fill="none"
              stroke="currentColor"
              strokeWidth={2.5}
              strokeLinecap="round"
            >
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="space-y-4 p-6">
          <div
            className="border-l-2 pl-3"
            style={{ borderColor: cls.placeholderColor }}
          >
            <p className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted-foreground">
              What it is
            </p>
            <p className="mt-1 text-sm leading-relaxed text-foreground/80">
              {cls.description}
            </p>
          </div>

          <div
            className="border-l-2 pl-3"
            style={{ borderColor: cls.placeholderColor }}
          >
            <p className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted-foreground">
              Why it matters
            </p>
            <p className="mt-1 text-sm leading-relaxed text-foreground/80">
              {cls.why}
            </p>
          </div>

          <div
            className="border-l-2 pl-3"
            style={{ borderColor: cls.placeholderColor }}
          >
            <p className="text-[11px] font-semibold uppercase tracking-[0.15em] text-muted-foreground">
              Key symptoms
            </p>
            <ul className="mt-1.5 space-y-1">
              {cls.symptoms.map((s) => (
                <li
                  key={s}
                  className="flex items-center gap-2 text-sm text-foreground/80"
                >
                  <span
                    className="h-1.5 w-1.5 shrink-0 rounded-full"
                    style={{ backgroundColor: cls.placeholderColor }}
                  />
                  {s}
                </li>
              ))}
            </ul>
          </div>

          <a
            href={cls.wiki}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium text-white shadow-sm hover:shadow"
            style={{ background: cls.gradient }}
          >
            Read more on Wikipedia
            <svg
              viewBox="0 0 24 24"
              className="h-4 w-4"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </a>
        </div>
      </div>
    </div>
  );
}

// ── Domain gap section ───────────────────────────────────────────────────────

function DomainGapSection() {
  return (
    <div className="mx-auto max-w-5xl px-6 py-12">
      <div className="text-center">
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
          Honest reporting
        </p>
        <h2 className="mt-2 font-display text-2xl font-semibold sm:text-3xl">
          The domain gap
        </h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Same model, different conditions.
        </p>
      </div>

      {/* Stat cards */}
      <div className="mt-8 grid grid-cols-1 items-center gap-4 sm:grid-cols-[1fr_auto_1fr]">
        <div className="rounded-2xl border border-border bg-card p-6 text-center shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
            In-domain (controlled)
          </p>
          <p className="mt-3 font-display text-5xl font-semibold text-success">
            98.4<span className="text-3xl">%</span>
          </p>
          <p className="mt-2 text-xs text-muted-foreground">
            Held-out lab images &middot; macro-F1
          </p>
        </div>

        <div className="flex flex-col items-center gap-1 py-2">
          <svg
            viewBox="0 0 24 24"
            className="h-6 w-6 text-warning-foreground"
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <line x1="12" y1="5" x2="12" y2="19" />
            <polyline points="19 12 12 19 5 12" />
          </svg>
          <span className="rounded-full border border-border bg-card px-3 py-1 font-mono text-sm font-semibold text-foreground">
            -36.7 pts
          </span>
        </div>

        <div className="rounded-2xl border border-border bg-card p-6 text-center shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
            Field conditions
          </p>
          <p className="mt-3 font-display text-5xl font-semibold text-warning-foreground">
            61.7<span className="text-3xl">%</span>
          </p>
          <p className="mt-2 text-xs text-muted-foreground">
            Real-world phone photos &middot; macro-F1
          </p>
        </div>
      </div>

      {/* Per-class recall */}
      <div className="mt-8 rounded-2xl border border-border bg-card p-6 shadow-sm">
        <p className="mb-5 text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
          Per-class recall
        </p>
        <div className="space-y-5">
          {DOMAIN_GAP.perClass.map((row) => (
            <div key={row.name}>
              <p className="mb-2 text-sm font-medium">{row.name}</p>
              {/* Controlled bar */}
              <div className="flex items-center gap-3">
                <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-secondary">
                  <div
                    className="h-full rounded-full bg-success"
                    style={{ width: `${row.controlled * 100}%` }}
                  />
                </div>
                <div className="w-20 text-right">
                  <span className="font-mono text-xs font-medium">
                    {Math.round(row.controlled * 100)}%
                  </span>
                  <br />
                  <span className="text-[10px] text-muted-foreground">
                    controlled
                  </span>
                </div>
              </div>
              {/* Field bar */}
              <div className="mt-1 flex items-center gap-3">
                <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-secondary">
                  {row.field !== null && (
                    <div
                      className="h-full rounded-full bg-warning"
                      style={{ width: `${row.field * 100}%` }}
                    />
                  )}
                </div>
                <div className="w-20 text-right">
                  <span className="font-mono text-xs font-medium">
                    {row.field !== null
                      ? `${Math.round(row.field * 100)}%`
                      : "—"}
                  </span>
                  <br />
                  <span className="text-[10px] text-muted-foreground">
                    field
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="mt-5 flex items-center justify-end gap-4">
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className="h-2.5 w-2.5 rounded-full bg-success" />
            controlled
          </span>
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <span className="h-2.5 w-2.5 rounded-full bg-warning" />
            field
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Main app ─────────────────────────────────────────────────────────────────

export default function App() {
  const [dark, setDark] = useState(() => {
    const stored = localStorage.getItem("theme");
    if (stored) return stored === "dark";
    return false;
  });
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [infoClass, setInfoClass] = useState<(typeof CLASSES)[number] | null>(
    null,
  );
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);

  const handleFile = useCallback(
    (f: File) => {
      if (!f.type.startsWith("image/")) return;
      if (preview) URL.revokeObjectURL(preview);
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setPred(null);
    },
    [preview],
  );

  const [apiError, setApiError] = useState<string | null>(null);

  const classify = useCallback(async () => {
    if (!file) return;
    setLoading(true);
    setApiError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Server error" }));
        throw new Error(err.error || `HTTP ${res.status}`);
      }

      const result = await res.json();

      const probabilities: ClassProbability[] = CLASSES.map((c) => ({
        name: c.name,
        key: c.key,
        probability: result.probabilities[c.key] ?? 0,
      })).sort((a, b) => b.probability - a.probability);

      const confidence: number = result.confidence;
      const pick = CLASSES.find((c) => c.key === result.label)!;
      const risk: Prediction["risk"] =
        confidence > 0.8 ? "Low" : confidence > 0.6 ? "Moderate" : "High";

      setPred({
        label: pick.name,
        key: pick.key,
        confidence,
        probabilities,
        risk,
        needsReview: confidence < 0.65,
      });
    } catch (err) {
      setApiError(
        err instanceof Error ? err.message : "Failed to connect to model server",
      );
    } finally {
      setLoading(false);
    }
  }, [file]);

  const reset = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setPred(null);
  }, [preview]);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);

      const f = e.dataTransfer.files[0];
      if (f) {
        handleFile(f);
        return;
      }

      if (draggedSample) {
        const sample = draggedSample;
        draggedSample = null;
        fetch(sample.src)
          .then((r) => r.blob())
          .then((blob) => {
            handleFile(
              new File([blob], `${sample.classKey}_${sample.tag}.jpg`, {
                type: "image/jpeg",
              }),
            );
          });
      }
    },
    [handleFile],
  );

  return (
    <div className="flex min-h-screen flex-col">
      {/* ── Header ──────────────────────────────────────────────────── */}
      <header className="mx-auto flex w-full max-w-5xl items-center justify-between px-6 pt-8">
        <div className="flex items-center gap-3">
          <LeafMark />
          <div>
            <p className="font-display text-lg font-semibold leading-none">
              MaizeDetection
            </p>
            <p className="text-xs text-muted-foreground">
              Corn leaf disease classifier
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setDark(!dark)}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-border bg-card text-muted-foreground hover:border-primary/40 hover:text-foreground hover:shadow-sm"
            aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {dark ? (
              <svg
                viewBox="0 0 24 24"
                fill="none"
                className="h-4 w-4"
                stroke="currentColor"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="5" />
                <line x1="12" y1="1" x2="12" y2="3" />
                <line x1="12" y1="21" x2="12" y2="23" />
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                <line x1="1" y1="12" x2="3" y2="12" />
                <line x1="21" y1="12" x2="23" y2="12" />
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
              </svg>
            ) : (
              <svg
                viewBox="0 0 24 24"
                fill="none"
                className="h-4 w-4"
                stroke="currentColor"
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            )}
          </button>
          <a
            href="https://github.com/CaiZhengTech/MaizeDetection"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden items-center gap-1 rounded-full border border-border bg-card px-3.5 py-1.5 text-xs font-medium text-foreground hover:border-primary/40 hover:shadow-sm sm:inline-flex"
          >
            GitHub
            <span className="text-[10px]">&#8599;</span>
          </a>
        </div>
      </header>

      {/* ── Hero ────────────────────────────────────────────────────── */}
      <div className="mx-auto max-w-5xl px-6 pb-8 pt-12 text-center sm:pt-16">
        <h1 className="font-display text-4xl font-semibold sm:text-5xl">
          Drop a leaf,{" "}
          <span className="text-primary">see the diagnosis</span>
        </h1>
        <p className="mx-auto mt-4 max-w-xl text-sm text-muted-foreground sm:text-base">
          Experiment with a PyTorch / EfficientNet-B0 model trained on four corn
          foliar conditions. Upload any corn leaf image - the model will
          classify it and show its confidence.
        </p>
      </div>

      {/* ── Main grid ───────────────────────────────────────────────── */}
      <div className="mx-auto grid max-w-5xl gap-5 px-6 md:grid-cols-2">
        {/* Upload zone */}
        <div>
          <div
            className={`relative flex h-[460px] flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed bg-card transition-all ${
              dragging
                ? "scale-[1.005] border-primary bg-accent/30"
                : "border-border hover:border-primary/50"
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            {!preview ? (
              <div className="flex flex-col items-center gap-4 p-8">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-secondary text-primary shadow-sm">
                  <LeafIcon className="h-7 w-7" />
                </div>
                <div className="text-center">
                  <p className="font-display text-xl">
                    Drop a corn leaf image
                  </p>
                  <p className="mt-1.5 text-sm text-muted-foreground">
                    PNG, JPG, or WebP - or drag a sample below
                  </p>
                </div>
                <button
                  onClick={() => inputRef.current?.click()}
                  className="rounded-full bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground shadow-sm hover:bg-primary/90 hover:shadow"
                >
                  Choose image
                </button>
              </div>
            ) : (
              <div className="flex w-full flex-col items-center gap-3 p-5">
                <div className="relative w-full overflow-hidden rounded-xl border border-border bg-muted">
                  <img
                    src={preview}
                    alt="Uploaded leaf"
                    className="mx-auto block max-h-[320px] w-auto object-contain"
                  />
                </div>
                <p className="max-w-[240px] truncate text-xs text-muted-foreground">
                  {file?.name}
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={classify}
                    disabled={loading}
                    className="inline-flex items-center gap-2 rounded-full bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground shadow-sm hover:bg-primary/90 hover:shadow disabled:opacity-60"
                  >
                    {loading ? (
                      <>
                        <Spinner /> Classifying&hellip;
                      </>
                    ) : (
                      "Classify leaf"
                    )}
                  </button>
                  <button
                    onClick={reset}
                    className="rounded-full border border-border bg-card px-5 py-2.5 text-sm font-medium text-foreground hover:border-primary/40 hover:shadow-sm"
                  >
                    Try another
                  </button>
                </div>
              </div>
            )}
          </div>
          <p className="mt-3 text-center text-[11px] text-muted-foreground">
            For research and educational use only. Not a treatment
            recommendation.
          </p>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
              e.target.value = "";
            }}
          />
        </div>

        {/* Result card */}
        <div>
          <div className="flex h-[460px] flex-col overflow-y-auto rounded-2xl border border-border bg-card p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-[0.15em] text-muted-foreground">
                Result
              </span>
              <span className="font-mono text-[10px] text-muted-foreground">
                POST /predict
              </span>
            </div>

            {apiError && (
              <div className="mt-3 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                <p className="font-medium">Connection failed</p>
                <p className="mt-1 text-xs opacity-80">{apiError}</p>
                <p className="mt-2 text-xs opacity-60">
                  Start the model server: <code className="font-mono">python serve.py</code>
                </p>
              </div>
            )}

            {!pred && !loading ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-3 text-center">
                <div className="h-10 w-10 rounded-full bg-secondary" />
                <p className="max-w-[200px] text-sm text-muted-foreground">
                  Upload an image and tap{" "}
                  <span className="font-medium text-foreground">Classify</span>{" "}
                  to see the prediction.
                </p>
              </div>
            ) : loading ? (
              <div className="flex flex-1 flex-col items-center justify-center gap-3">
                <Spinner />
                <p className="text-sm text-muted-foreground">
                  Running model&hellip;
                </p>
              </div>
            ) : pred ? (
              <div className="mt-4 flex flex-1 flex-col">
                <h2 className="font-display text-3xl leading-tight">
                  {pred.label}
                </h2>
                <p className="mt-0.5 font-mono text-[11px] text-muted-foreground">
                  {pred.key}
                </p>

                <div className="mt-5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">
                      Confidence
                    </span>
                    <span className="font-mono text-sm font-medium">
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="mt-1.5 h-2 w-full overflow-hidden rounded-full bg-secondary">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-leaf to-corn transition-all duration-700"
                      style={{ width: `${pred.confidence * 100}%` }}
                    />
                  </div>
                </div>

                <div className="mt-3 flex flex-wrap gap-1.5">
                  <Badge
                    tone={
                      pred.risk === "Low"
                        ? "success"
                        : pred.risk === "Moderate"
                          ? "warning"
                          : "destructive"
                    }
                  >
                    {pred.risk} risk
                  </Badge>
                  {pred.needsReview && (
                    <Badge tone="warning">Human review recommended</Badge>
                  )}
                </div>

                <div className="mt-5">
                  <p className="mb-2 text-xs text-muted-foreground">
                    All class probabilities
                  </p>
                  <div className="space-y-3">
                    {pred.probabilities.map((p) => (
                      <div key={p.key}>
                        <div className="mb-1 flex items-center justify-between">
                          <span className="truncate text-xs font-medium">
                            {p.name}
                          </span>
                          <span className="ml-2 font-mono text-xs tabular-nums text-muted-foreground">
                            {(p.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-3 w-full overflow-hidden rounded-full bg-secondary">
                          <div
                            className="h-full rounded-full bg-primary/70 transition-all duration-500"
                            style={{ width: `${p.probability * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {/* ── Sample images ──────────────────────────────────────────── */}
      <div className="mx-auto max-w-5xl px-6 pt-8">
        <div className="mb-4 flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            Sample images
          </p>
          <p className="text-xs text-muted-foreground">
            drag - drop above
          </p>
        </div>
        <div className="flex flex-wrap justify-center gap-2">
          {SAMPLE_IMAGES.map((sample) => {
            const cls = CLASSES.find((c) => c.key === sample.classKey)!;
            return (
              <div
                key={sample.id}
                draggable
                onDragStart={(e) => {
                  draggedSample = sample;
                  e.dataTransfer.effectAllowed = "copy";
                  e.dataTransfer.setData("text/plain", sample.id);
                }}
                onDragEnd={() => {
                  draggedSample = null;
                }}
                className="w-[112px] shrink-0 cursor-grab overflow-hidden rounded-xl border border-border bg-card shadow-sm transition-all hover:border-primary/40 hover:shadow active:cursor-grabbing"
              >
                <div
                  className="relative h-20 overflow-hidden"
                  style={{ backgroundColor: cls.placeholderColor }}
                >
                  <img
                    src={sample.src}
                    alt={sample.label}
                    className="h-full w-full object-cover"
                    draggable={false}
                  />
                  <span
                    className={`absolute right-1 top-1 rounded px-1.5 py-0.5 text-[8px] font-bold uppercase ${
                      sample.tag === "field"
                        ? "bg-warning text-white"
                        : "bg-success text-white"
                    }`}
                  >
                    {sample.tag}
                  </span>
                </div>
                <p className="px-2 py-2 text-center text-[11px] font-medium leading-tight">
                  {sample.label}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Domain gap ─────────────────────────────────────────────── */}
      <DomainGapSection />

      {/* ── Supported classes ───────────────────────────────────────── */}
      <div className="mx-auto max-w-5xl px-6 pb-12">
        <p className="mb-4 text-center text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
          Supported classes
        </p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {CLASSES.map((cls) => (
            <button
              key={cls.key}
              onClick={() => setInfoClass(cls)}
              className="cursor-pointer overflow-hidden rounded-xl border border-border bg-card text-center transition-all hover:border-primary/40 hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background"
            >
              <div
                className="h-1.5"
                style={{ background: cls.gradient }}
              />
              <div className="px-4 py-3">
                <p className="font-display text-sm">{cls.name}</p>
                <p className="font-mono text-[10px] text-muted-foreground">
                  {cls.key}
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="mx-auto mt-auto w-full max-w-5xl px-6 pb-8 text-center">
        <p className="text-xs text-muted-foreground">
          Built with Python, PyTorch, EfficientNet-B0, Scikit-Learn, Pandas,
          NumPy &amp; Jupyter.
        </p>
        <div className="mt-2 flex items-center justify-center gap-3 text-xs text-muted-foreground">
          <a
            href="https://github.com/CaiZhengTech/MaizeDetection"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-foreground"
          >
            GitHub
          </a>
          <span>&middot;</span>
          <a
            href="https://github.com/CaiZhengTech/MaizeDetection#readme"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-foreground"
          >
            README
          </a>
          <span>&middot;</span>
          <a href="https://caizhengtech.com/" className="hover:text-foreground">
            Portfolio
          </a>
        </div>
      </footer>

      {/* ── Class info dialog ───────────────────────────────────────── */}
      {infoClass && (
        <ClassInfoDialog cls={infoClass} onClose={() => setInfoClass(null)} />
      )}
    </div>
  );
}
