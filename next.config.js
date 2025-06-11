/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://turing-test.featurely.ai/:path*',
      },
    ];
  },
};

module.exports = nextConfig; 