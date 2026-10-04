import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Shell } from "@/components/Shell";
import { SystemProvider } from "@/components/System";
import { ToastProvider } from "@/components/Toast";

export const metadata: Metadata = {
  title: "CLIPFORGE AI",
  description: "AI clipping studio: long videos in, captioned vertical shorts out.",
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
          <SystemProvider>
            <Shell>{children}</Shell>
          </SystemProvider>
        </ToastProvider>
      </body>
    </html>
  );
}
