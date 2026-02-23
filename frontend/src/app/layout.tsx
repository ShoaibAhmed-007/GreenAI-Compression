import type { Metadata } from 'next';
import './globals.css';

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
    <html lang="en">
      <body className="min-h-screen bg-gray-50">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-green-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">G</span>
                </div>
                <div>
                  <h1 className="text-lg font-bold text-gray-900">GreenAI</h1>
                  <p className="text-xs text-gray-500 -mt-0.5">Model Compression for Edge Devices</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <span className="badge-green">FYP 2025-26</span>
                <a
                  href="http://localhost:8000/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-gray-500 hover:text-green-600 transition-colors"
                >
                  API Docs
                </a>
              </div>
            </div>
          </div>
        </header>

        {/* Main content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-gray-200 bg-white mt-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <p className="text-center text-sm text-gray-500">
              GreenAI — Final Year Project | 11 Pretrained Models · CIFAR-10/100 |
              Pruning · Quantization · Hybrid · Knowledge Distillation
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
