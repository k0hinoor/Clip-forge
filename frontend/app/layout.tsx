import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Shell } from "@/components/Shell";
import { ToastProvider } from "@/components/Toast";

export const metadata: Metadata = {
  title: "CLIPFORGE AI",
  description: "Local AI clipping studio: long videos in, vertical shorts out. Everything runs on this machine.",
};

export const viewport: Viewport = {
  themeColor: "#07090f",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ToastProvider>
          <Shell>{children}</Shell>
        </ToastProvider>
      </body>
    </html>
  );
}
