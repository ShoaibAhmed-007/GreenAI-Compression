'use client';

import { useState, useCallback, useEffect, useRef, createContext, useContext } from 'react';

// ── Search + Sidebar Context ─────────────────────────────────────────────────

interface AppContextValue {
  query: string;
  setQuery: (q: string) => void;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

const AppContext = createContext<AppContextValue>({
  query: '',
  setQuery: () => { },
  sidebarOpen: false,
  setSidebarOpen: () => { },
});

export function useSearch() {
  const { query, setQuery } = useContext(AppContext);
  return { query, setQuery };
}

export function useSidebar() {
  const { sidebarOpen, setSidebarOpen } = useContext(AppContext);
  return { sidebarOpen, setSidebarOpen };
}

export function SearchProvider({ children }: { children: React.ReactNode }) {
  const [query, setQuery] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  return (
    <AppContext.Provider value={{ query, setQuery, sidebarOpen, setSidebarOpen }}>
      {children}
    </AppContext.Provider>
  );
}

// ── Helper: debounce ────────────────────────────────────────────────────────
function useDebounce(value: string, delay: number) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const handler = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(handler);
  }, [value, delay]);
  return debounced;
}

// ── Mobile Sidebar Drawer ───────────────────────────────────────────────────
export function MobileSidebar({ children }: { children: React.ReactNode }) {
  const { sidebarOpen, setSidebarOpen } = useSidebar();

  // Close on Escape key
  useEffect(() => {
    if (!sidebarOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSidebarOpen(false);
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [sidebarOpen, setSidebarOpen]);

  // Prevent body scroll when open
  useEffect(() => {
    document.body.style.overflow = sidebarOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [sidebarOpen]);

  return (
    <>
      {/* Backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm md:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Drawer */}
      <div
        className={`
          fixed left-0 top-0 h-screen w-64 z-40 md:hidden
          bg-surface-container-lowest flex flex-col py-4 space-y-2
          transition-transform duration-300 ease-in-out
          ${sidebarOpen ? 'translate-x-0 shadow-2xl' : '-translate-x-full'}
        `}
        aria-label="Mobile navigation"
      >
        {/* Close button inside drawer */}
        <button
          type="button"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close navigation"
          className="absolute top-4 right-4 text-on-surface-variant hover:text-on-surface p-1 rounded-lg transition-colors"
        >
          <span className="material-symbols-outlined text-xl leading-none">close</span>
        </button>

        {children}
      </div>
    </>
  );
}

// ── TopNavbar ───────────────────────────────────────────────────────────────
export function TopNavbar() {
  const [inputValue, setInputValue] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { setQuery } = useSearch();
  const { setSidebarOpen } = useSidebar();

  const debouncedValue = useDebounce(inputValue, 250);

  // Propagate debounced value to the search context
  useEffect(() => {
    setQuery(debouncedValue);
  }, [debouncedValue, setQuery]);

  const handleClear = useCallback(() => {
    setInputValue('');
    setQuery('');
    inputRef.current?.focus();
  }, [setQuery]);

  const handleSearchIconClick = useCallback(() => {
    inputRef.current?.focus();
  }, []);

  return (
    <header className="fixed top-0 right-0 left-0 md:left-64 z-50 bg-emerald-950/60 backdrop-blur-xl shadow-2xl shadow-emerald-950/50 flex justify-between items-center px-6 py-3">
      <div className="flex items-center gap-4">
        {/* Mobile menu toggle — opens the sidebar drawer */}
        <button
          type="button"
          onClick={() => setSidebarOpen(true)}
          aria-label="Open navigation menu"
          className="text-primary md:hidden p-1 rounded-lg hover:bg-primary/10 transition-colors active:scale-90"
        >
          <span className="material-symbols-outlined">menu</span>
        </button>

        {/* Search bar */}
        <div className="relative hidden sm:block">
          {/* Search icon — clickable to focus the input */}
          <button
            type="button"
            onClick={handleSearchIconClick}
            aria-label="Focus search"
            className="absolute left-3 top-1/2 -translate-y-1/2 text-emerald-100/40 hover:text-emerald-100/70 transition-colors"
          >
            <span className="material-symbols-outlined text-sm leading-none">search</span>
          </button>

          <input
            ref={inputRef}
            id="navbar-search"
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Search models"
            aria-label="Search models"
            className={`
              bg-emerald-900/20 border border-transparent rounded-full
              pl-10 pr-8 py-1.5 text-sm text-emerald-100
              placeholder:text-emerald-100/40
              focus:ring-2 focus:ring-primary focus:border-primary/40
              w-64 transition-all outline-none
              ${isFocused ? 'w-80' : ''}
            `}
          />

          {/* Clear button — visible only when there is text */}
          {inputValue && (
            <button
              type="button"
              onClick={handleClear}
              aria-label="Clear search"
              className="absolute right-3 top-1/2 -translate-y-1/2 text-emerald-100/40 hover:text-emerald-100/80 transition-colors"
            >
              <span className="material-symbols-outlined text-sm leading-none">close</span>
            </button>
          )}
        </div>
      </div>

      <div className="flex items-center gap-6">
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="text-emerald-100/60 hover:text-emerald-100 text-sm font-medium transition-all"
        >
          API Docs
        </a>
        <button className="relative text-emerald-100/60 hover:text-emerald-100 transition-all active:scale-90">
          <span className="material-symbols-outlined">notifications</span>
          <span className="absolute top-0 right-0 w-2 h-2 bg-primary rounded-full border-2 border-emerald-950" />
        </button>
        <div className="w-8 h-8 rounded-full overflow-hidden bg-primary/20 flex items-center justify-center cursor-pointer hover:scale-95 transition-transform">
          <span className="text-primary font-bold text-sm">G</span>
        </div>
      </div>
    </header>
  );
}
