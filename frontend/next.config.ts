/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'i.pinimg.com',
        pathname: '/originals/**',
      },
    ],
  },
  async rewrites() {
    const base = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:5000';
    return [
      {
        source: '/api/:path*',
        destination: `${base}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
