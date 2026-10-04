/** @type {import('next').NextConfig} */
const API_TARGET = (process.env.CLIPFORGE_API ?? "http://127.0.0.1:8317").replace(/\/+$/, "");

const nextConfig = {
  reactStrictMode: true,
  // Everything the browser calls goes through this proxy, so the UI and the API
  // share one origin (no CORS) and the backend never has to be public.
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API_TARGET}/api/:path*` },
      { source: "/media/:path*", destination: `${API_TARGET}/media/:path*` },
    ];
  },
  experimental: {
    // The proxy's default 30 s timeout cut long requests (large uploads,
    // storage cleanup) and idle event streams. The event stream sends a
    // heartbeat every 15 s; this only bounds genuinely stuck requests.
    proxyTimeout: 10 * 60 * 1000,
  },
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: false },
};

export default nextConfig;
