import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MedLingua — Understand your medical reports",
  description: "Extract, summarize, and translate medical reports in a private workspace.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
