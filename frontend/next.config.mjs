/** @type {import('next').NextConfig} */
const API_TARGET = process.env.CLIPFORGE_API ?? "http://127.0.0.1:8317";

const nextConfig = {
  reactStrictMode: true,
  // The UI and the API run side by side on the same machine. Everything the
  // browser calls goes through this proxy so there is only ever one origin, and
  // the backend is never exposed beyond localhost.
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API_TARGET}/api/:path*` },
      { source: "/media/:path*", destination: `${API_TARGET}/media/:path*` },
    ];
  },
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: false },
};

export default nextConfig;
