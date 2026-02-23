import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Handle Chrome DevTools probe (returns 204 No Content)
  if (pathname.includes('/.well-known/appspecific/com.chrome.devtools.json')) {
    return new NextResponse(null, { status: 204 });
  }

  // Redirect /Pages/index to home (likely browser extension/crawler)
  if (pathname === '/Pages/index' || pathname.startsWith('/Pages/')) {
    return NextResponse.redirect(new URL('/', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
};
