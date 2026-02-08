import StreamlitEmbed from "@/components/StreamlitEmbed";

function LinkCard({
  title,
  description,
  href,
  badge,
}: {
  title: string;
  description: string;
  href: string;
  badge?: string;
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="block rounded-xl border border-gray-200 bg-white p-6 shadow-sm hover:shadow-md transition"
    >
      <div className="flex items-center justify-between gap-4">
        <h2 className="text-2xl font-bold text-gray-900">{title}</h2>
        {badge ? (
          <span className="inline-flex items-center rounded-full bg-yellow-100 px-3 py-1 text-sm font-semibold text-yellow-900">
            {badge}
          </span>
        ) : null}
      </div>
      <p className="mt-3 text-gray-700">{description}</p>
      <p className="mt-4 text-sm font-semibold text-blue-700">
        View on GitHub →
      </p>
    </a>
  );
}

export default function ProjectsPage() {
  return (
    <main className="max-w-5xl mx-auto px-6 py-12">
      <h1 className="text-4xl font-extrabold mb-10 text-gray-900">
        Portfolio Projects
      </h1>

      {/* p01 */}
      <StreamlitEmbed
        title="Customer Churn Prediction Dashboard"
        description="An interactive machine learning dashboard that predicts customer churn and visualizes key insights using Streamlit."
        url="https://ja-portfolio-churn-dashboard.streamlit.app"
      />

      {/* p02 */}
      <StreamlitEmbed
        title="Financial Fraud Detection (Snowflake + dbt)"
        description="A Snowflake + dbt-powered data pipeline that detects and visualizes financial fraud patterns with analytics and lineage insights."
        url="https://ja-portfolio-snowf-dbt-showcase.streamlit.app"
      />

      {/* p03 */}
      <StreamlitEmbed
        title="NLP Resume Analyzer"
        description="A natural language processing app that analyzes resumes, extracts key entities, and evaluates alignment with job descriptions."
        url="https://ja-portfolio-nlp-resume.streamlit.app"
      />

      {/* p04 — WIP */}
      <div className="mt-8">
        <LinkCard
          title="Multimodal Search"
          badge="WIP"
          description="Search images using text or image similarity (embeddings + FAISS). The local prototype is complete; deployment refactor is currently in progress."
          href="https://github.com/jomana-e/portfolio-2025/blob/main/p04_multimodal_search/README.md"
        />
      </div>
    </main>
  );
}
