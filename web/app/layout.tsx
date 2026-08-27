import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Neural Estate — Rete neurale interattiva",
  description: "Esplora una rete neurale Java e sperimenta una stima immobiliare in tempo reale.",
  metadataBase: new URL("https://gianlucabove.it"),
  alternates: { canonical: "/neural-network/" },
  openGraph: {
    title: "Neural Estate",
    description: "Dentro una rete neurale, un neurone alla volta.",
    url: "/neural-network/",
    siteName: "Gianluca Bove",
    locale: "it_IT",
    type: "website",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="it">
      <body>{children}</body>
    </html>
  );
}
