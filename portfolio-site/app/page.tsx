export default function Home() {
  const projects = [
    {
      title: "Predictive Customer Churn Dashboard",
      description:
        "A full-stack ML app predicting telecom churn using Streamlit and scikit-learn.",
      link: "https://ja-portfolio-churn-dashboard.streamlit.app/",
      tags: ["Machine Learning", "Streamlit", "scikit-learn"],
      type: "streamlit",
    },
    {
      title: "Fraud Detection with Snowflake + dbt",
      description:
        "End-to-end data pipeline using Snowflake, dbt, and SQL-based feature engineering.",
      link: "https://ja-portfolio-snowf-dbt-showcase.streamlit.app/",
      tags: ["Data Engineering", "Snowflake", "dbt"],
      type: "streamlit",
    },
    {
      title: "NLP Resume & JD Analyzer",
      description:
        "A semantic analyzer that matches resumes with job descriptions using embeddings, taxonomy coverage, and KeyBERT-powered insights.",
      link: "https://ja-portfolio-nlp-resume-analyzer.streamlit.app/",
      tags: ["NLP", "Hugging Face", "Streamlit"],
      type: "streamlit",
    },
    {
      title: "Multimodal Image Search (WIP)",
      description:
        "Search images using text or image similarity (embeddings + FAISS). Local prototype complete; deployment refactor in progress.",
      link: "https://github.com/jomana-e/portfolio-2025/blob/main/p04_multimodal_search/README.md",
      tags: ["Multimodal", "Vector Search", "FAISS", "WIP"],
      type: "github",
    },
  ];

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 px-6 py-12">
      <div className="max-w-5xl mx-auto">
        <h1 className="text-5xl font-bold mb-10 text-center">
          Data Science & AI Portfolio
        </h1>

        <div className="grid md:grid-cols-2 gap-8">
          {projects.map((p) => (
            <div
              key={p.title}
              className="bg-gray-900 border border-gray-800 p-6 rounded-2xl shadow-lg hover:shadow-2xl hover:border-blue-400 transition"
            >
              <h2 className="text-2xl font-semibold mb-2">{p.title}</h2>
              <p className="text-gray-400 mb-4">{p.description}</p>

              <div className="flex flex-wrap gap-2 mb-4">
                {p.tags.map((tag) => (
                  <span
                    key={tag}
                    className="bg-gray-800 text-sm text-blue-300 px-2 py-1 rounded-md"
                  >
                    {tag}
                  </span>
                ))}
              </div>

              <a
                href={p.link}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-300 hover:translate-x-1 transition-transform font-medium"
              >
                {p.type === "streamlit"
                  ? "→ View Project in Streamlit"
                  : "→ View Project on GitHub (WIP)"}
              </a>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
