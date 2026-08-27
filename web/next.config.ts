import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/neural-network",
  assetPrefix: "/neural-network",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
