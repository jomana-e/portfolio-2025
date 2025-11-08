/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/churn-dashboard/:path*',
        destination: 'https://ja-portfolio-churn-dashboard.streamlit.app/:path*',
      },
      {
        source: '/fraud-dbt/:path*',
        destination: 'https://ja-portfolio-snowf-dbt-showcase.streamlit.app/:path*',
      },
    ];
  },
};

export default nextConfig;
