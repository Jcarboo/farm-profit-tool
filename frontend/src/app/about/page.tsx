// frontend/app/about/page.tsx
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About | Farm Profit Estimator",
  description:
    "What this tool does, how the gradient-descent model works, and how to interpret results.",
};

const greenPanel =
  "bg-emerald-900/30 border border-emerald-700/60 rounded-2xl";

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-emerald-950 to-green-900 text-green-50">
      <div className="max-w-5xl mx-auto px-5 py-10 space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight">
            About the Farm Profit Estimator
          </h1>
          <p className="text-green-200/90">
            A practical tool to estimate year-end net returns from farm
            operating costs, with an emphasis on clarity and actionability,
          </p>
          <h3 className='text-teal-500'>Made by Jayson Carboo</h3>
        </header>

        <section className={`${greenPanel} p-5 space-y-3`}>
          <h2 className="text-xl font-semibold">What it does</h2>
          <ul className="list-disc pl-5 text-green-100/90 space-y-1">
            <li>
              Trains a separate model for each crop using historical USDA Cost
              &amp; Returns data.
            </li>
            <li>
              Accepts spending by category —{" "}
              <span className="font-semibold">
                Seed, Fertilizer, Chemicals, Services, FLE, Repairs, Water,
                Interest
              </span>{" "}
              — and predicts <span className="font-semibold">net value</span>.
            </li>
            <li>
              Supports multi-crop scenarios
            </li>
            <li>
              Displays how each crop contributes to the overall profit.
            </li>
            <li>
              Includes a live <span className="font-semibold">What-If</span> panel to
              adjust spending and see real-time changes in the prediction.
            </li>
            <li>
              The project expands upon my CMSC320 final project —
              details available{" "}
              <a
                href="https://jcarboo.github.io"
                className="text-red-400 underline"
              >
                here
              </a>
              .
            </li>
          </ul>
        </section>

        <section className={`${greenPanel} p-5 space-y-3`}>
          <h2 className="text-xl font-semibold">Data pipeline</h2>
          <p className="text-green-100/90">
            <a 
                href="https://www.ers.usda.gov/data-products/commodity-costs-and-returns" 
                className="text-red-400 underline"
            >
                Source: USDA Cost &amp; Returns data per crop (Region, Year).</a>
          </p>
          <ul className="list-disc pl-5 text-green-100/90 space-y-1">
            <li>Normalizes item names (e.g., “Fuel, lube, and electricity” → FLE).</li>
            <li>Removes totals and U.S. aggregates; keeps Region-level rows.</li>
            <li>
              Pivots to a wide table; the 8 operating cost categories are the{" "}
              <span className="font-semibold">features</span>, and{" "}
              <span className="font-semibold">net value</span> is the target.
            </li>
            <li>
              By default, trains on <span className="font-semibold">all historical rows</span>{" "}
              for the selected crop (matching the original research setup).
            </li>
          </ul>
        </section>

        <section className={`${greenPanel} p-5 space-y-3`}>
          <h2 className="text-xl font-semibold">Modeling approach</h2>
          <ul className="list-disc pl-5 text-green-100/90 space-y-1">
            <li>
              <span className="font-semibold">Features:</span> the 8 cost categories
              standardized per crop (z-scores) plus a bias term.
            </li>
            <li>
              <span className="font-semibold">Model:</span> linear regression trained
              with <span className="font-semibold">batch gradient descent</span>; it
              minimizes <code>0.5 ∥Xθ − y∥²</code> on standardized inputs.
            </li>
            <li>
              <span className="font-semibold">Hyperparameters:</span>{" "}
              iterations <code>T = 1000</code>, learning rate <code>α = 0.001</code>.
              
            </li>
            <li>
              <span className="font-semibold">Interpretation:</span> larger-magnitude
              weights indicate stronger association with profit; positive weights
              mean higher spending tends to increase profit,
              and negative weights the opposite.
            </li>
            <li>
              <span className="font-semibold">Explainability:</span> per-feature{" "}
              <em>contributions</em> (β×z) and <em>marginal profit per dollar</em> (β/σ)
              are reported to clarify which categories move the prediction most.
            </li>
          </ul>
        </section>

       

        <section className={`${greenPanel} p-5 space-y-3`}>
          <h2 className="text-xl font-semibold">Limitations &amp; good usage</h2>
          <ul className="list-disc pl-5 text-green-100/90 space-y-1">
            <li>
              This project was made to aid decision making and should NOT be
              used as a sole source of financial advice (I am not a farmer and have no
              education in agriculture).
            </li>
            <li>
              Feature relationships with profit can change with weather, markets, and policy; historical
              patterns may not always hold, so it's important to take into account features that the model
              does not include.
            </li>
            <li>
              The model is linear; non-linear effects (e.g., diminishing returns)
              are not explicitly modeled in this version.
            </li>
          </ul>
        </section>

        <footer className="text-xs text-green-200/60 pb-10">
          Data © USDA. Estimates are for decision support and do not constitute financial advice.
        </footer>
      </div>
    </main>
  );
}
