import { HttpInterceptorFn } from '@angular/common/http';
import { inject, Injectable, signal } from '@angular/core';

const KEY = 'admin_token';

/** Holds the admin token for /sync endpoints. Kept in sessionStorage so it ends with the browser tab. */
@Injectable({ providedIn: 'root' })
export class AuthService {
  readonly token = signal(sessionStorage.getItem(KEY) ?? '');

  set(value: string): void {
    const v = value.trim();
    this.token.set(v);
    if (v) sessionStorage.setItem(KEY, v);
    else sessionStorage.removeItem(KEY);
  }
}

export const adminTokenInterceptor: HttpInterceptorFn = (req, next) => {
  const token = inject(AuthService).token();
  return next(token && req.url.includes('/sync/') ? req.clone({ setHeaders: { 'X-Admin-Token': token } }) : req);
};
