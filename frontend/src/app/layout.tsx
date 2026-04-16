import type { Metadata } from 'next';
import './globals.css';
import Link from 'next/link';
import { TopNavbar, SearchProvider, MobileSidebar } from '@/components/TopNavbar';

export const metadata: Metadata = {
  title: 'GreenAI — Model Compression Dashboard',
  description: 'Compress deep learning models for edge devices. Save energy. Reduce carbon emissions.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap"
          rel="stylesheet"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen bg-surface text-on-surface">
        <SearchProvider>
        {/* ── Sidebar (desktop) ── */}
        <aside className="h-screen w-64 fixed left-0 top-0 bg-surface-container-lowest flex flex-col py-4 space-y-2 z-40 hidden md:flex">
          <div className="px-6 mb-8">
            <h1 className="text-emerald-500 font-bold text-2xl font-headline tracking-tighter">
              GreenAI
            </h1>
            <p className="text-[0.65rem] text-on-surface-variant/50 font-technical uppercase tracking-widest">
              Sustainable Edge Intelligence
            </p>
          </div>
          <nav className="flex-1 space-y-1">
            <Link
              href="/"
              className="flex items-center gap-3 px-6 py-3 text-on-surface-variant hover:text-primary hover:bg-primary/5 rounded-lg mx-2 transition-colors duration-200"
            >
              <span className="material-symbols-outlined">dashboard</span>
              <span className="font-medium">Dashboard</span>
            </Link>
            <Link
              href="/model-comparison"
              className="flex items-center gap-3 px-6 py-3 text-on-surface-variant hover:text-primary hover:bg-primary/5 rounded-lg mx-2 transition-colors duration-200"
            >
              <span className="material-symbols-outlined">compare_arrows</span>
              <span className="font-medium">Model Comparison</span>
            </Link>
          </nav>
          <div className="mt-auto px-4">
            <div className="bg-primary/5 p-4 rounded-xl ghost-border text-center">
              <p className="text-xs text-primary/60 font-technical mb-2">FYP 2025-26</p>
              <div className="w-10 h-10 rounded-full bg-primary/20 mx-auto flex items-center justify-center">
                <span className="material-symbols-outlined text-primary">eco</span>
              </div>
            </div>
          </div>
        </aside>

        {/* ── Main wrapper ── */}
        <div className="md:ml-64 min-h-screen flex flex-col">
          {/* ── Top Nav Bar (client component with live search) ── */}
          <TopNavbar />

          {/* ── Main content ── */}
          <main className="pt-20 px-6 lg:px-8 pb-12 flex-1">
            {children}
          </main>

          {/* ── Footer ── */}
          <footer className="w-full py-8 mt-auto bg-surface-container-lowest border-t border-outline-variant/10">
            <div className="max-w-7xl mx-auto px-8 flex flex-col items-center space-y-4 text-center">
              <div className="text-primary font-semibold font-headline">GreenAI</div>
              <div className="flex gap-6">
                <span className="text-on-surface-variant/40 hover:text-primary text-xs transition-colors cursor-default">
                  Models
                </span>
                <span className="text-on-surface-variant/40 hover:text-primary text-xs transition-colors cursor-default">
                  Datasets
                </span>
                <span className="text-on-surface-variant/40 hover:text-primary text-xs transition-colors cursor-default">
                  Compression Techniques
                </span>
              </div>
              <p className="text-[0.65rem] text-on-surface-variant/30 font-technical uppercase tracking-widest">
                GreenAI — Final Year Project | 11 Pretrained Models · CIFAR-10/100 |
                Pruning · Quantization · Hybrid · Knowledge Distillation
              </p>
            </div>
          </footer>
        </div>

        {/* ── Mobile Sidebar Drawer (slides in from left on small screens) ── */}
        <MobileSidebar>
          <div className="px-6 mb-8">
            <h1 className="text-emerald-500 font-bold text-2xl font-headline tracking-tighter">
              GreenAI
            </h1>
            <p className="text-[0.65rem] text-on-surface-variant/50 font-technical uppercase tracking-widest">
              Sustainable Edge Intelligence
            </p>
          </div>
          <nav className="flex-1 space-y-1">
            <Link
              href="/"
              className="flex items-center gap-3 px-6 py-3 text-on-surface-variant hover:text-primary hover:bg-primary/5 rounded-lg mx-2 transition-colors duration-200"
            >
              <span className="material-symbols-outlined">dashboard</span>
              <span className="font-medium">Dashboard</span>
            </Link>
            <Link
              href="/model-comparison"
              className="flex items-center gap-3 px-6 py-3 text-on-surface-variant hover:text-primary hover:bg-primary/5 rounded-lg mx-2 transition-colors duration-200"
            >
              <span className="material-symbols-outlined">compare_arrows</span>
              <span className="font-medium">Model Comparison</span>
            </Link>
          </nav>
          <div className="mt-auto px-4">
            <div className="bg-primary/5 p-4 rounded-xl ghost-border text-center">
              <p className="text-xs text-primary/60 font-technical mb-2">FYP 2025-26</p>
              <div className="w-10 h-10 rounded-full bg-primary/20 mx-auto flex items-center justify-center">
                <span className="material-symbols-outlined text-primary">eco</span>
              </div>
            </div>
          </div>
        </MobileSidebar>

        {/* ── Mobile Bottom Nav Bar ── */}
        <nav className="md:hidden fixed bottom-0 left-0 right-0 glass-panel border-t border-outline-variant/20 px-6 py-3 flex justify-around items-center z-50">
          <Link href="/" className="flex flex-col items-center gap-1 text-primary">
            <span className="material-symbols-outlined">dashboard</span>
            <span className="text-[10px] font-bold">Dashboard</span>
          </Link>
          <Link href="/model-comparison" className="flex flex-col items-center gap-1 text-on-surface-variant/50">
            <span className="material-symbols-outlined">compare_arrows</span>
            <span className="text-[10px] font-bold">Compare</span>
          </Link>
        </nav>
        </SearchProvider>
      </body>
    </html>
  );
}
