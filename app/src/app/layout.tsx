import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "EchoSelf — Your Voice, Echoed Digitally",
  description:
    "A local-first AI voice identity platform. Enroll your voice, generate speech, and create your digital twin — entirely on-device.",
  openGraph: {
    title: "EchoSelf",
    description: "Your voice and identity, echoed digitally.",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700&family=JetBrains+Mono:wght@300;400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
