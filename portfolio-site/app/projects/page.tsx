import StreamlitEmbed from "@/components/StreamlitEmbed";

export default function ProjectsPage() {
  return (
    <main className="max-w-5xl mx-auto px-6 py-12">
      <h1 className="text-4xl font-extrabold mb-10 text-gray-900">
        Portfolio Projects
      </h1>

      <StreamlitEmbed
        title="Customer Churn Prediction Dashboard"
        description="An interactive machine learning dashboard that predicts customer churn and visualizes key insights using Streamlit."
        url="https://ja-portfolio-churn-dashboard.streamlit.app"
      />

      <StreamlitEmbed
        title="Financial Fraud Detection (Snowflake + dbt)"
        description="A Snowflake + dbt-powered data pipeline that detects and visualizes financial fraud patterns with analytics and lineage insights."
        url="https://ja-portfolio-snowf-dbt-showcase.streamlit.app"
      />
    </main>
  );
}
