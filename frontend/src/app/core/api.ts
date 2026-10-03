import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { catchError, map, Observable, of } from 'rxjs';

import {
  Categories, FailedImage, Job, ScopeList, ScopePreview, SearchParams, SearchResponse, SyncScope, SyncStatus,
} from './models';

/** Base path of the FastAPI service: nginx proxies /api in production, proxy.conf.json in `ng serve`. */
const BASE = '/api';

/** Turn an HttpErrorResponse into a readable message (surfaces FastAPI's `detail`). */
export function errorMessage(err: unknown): string {
  if (!(err instanceof HttpErrorResponse)) return 'Unexpected error';
  if (err.status === 0) return 'Cannot reach the API';
  const detail = err.error?.detail;
  const text = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((d) => d.msg).join('; ') : '';
  return `${err.status}: ${text || err.statusText}`;
}

@Injectable({ providedIn: 'root' })
export class Api {
  private readonly http = inject(HttpClient);

  online(): Observable<boolean> {
    return this.http.get(`${BASE}/openapi.json`).pipe(map(() => true), catchError(() => of(false)));
  }

  categories(): Observable<Categories> {
    return this.http.get<Categories>(`${BASE}/categories`);
  }

  searchImage(file: File, p: SearchParams): Observable<SearchResponse> {
    const form = new FormData();
    form.append('file', file, file.name);
    form.append('top_k', String(p.top_k));
    form.append('mode', p.mode);
    for (const k of ['plain_gold', 'solitaire', 'valid', 'franchise'] as const) form.append(k, p[k]);
    if (p.category) form.append('category', p.category);
    if (p.item_codes.trim()) form.append('item_codes', p.item_codes.trim());
    if (p.min_score > 0) form.append('min_score', String(p.min_score));
    return this.http.post<SearchResponse>(`${BASE}/search/image`, form);
  }

  status(): Observable<SyncStatus> {
    return this.http.get<SyncStatus>(`${BASE}/sync/status`);
  }

  scope(): Observable<ScopeList> {
    return this.http.get<ScopeList>(`${BASE}/sync/scope`);
  }

  /** Add a scope's images to the live index (no alias switch, nothing already indexed is dropped). */
  startAdd(scope: SyncScope): Observable<{ job_id: string }> {
    return this.http.post<{ job_id: string }>(`${BASE}/sync/incremental`, { scope });
  }

  cancel(): Observable<{ job_id: string; status: string }> {
    return this.http.post<{ job_id: string; status: string }>(`${BASE}/sync/cancel`, {});
  }

  preview(scope: SyncScope): Observable<ScopePreview> {
    return this.http.post<ScopePreview>(`${BASE}/sync/preview`, scope);
  }

  startFull(scope: SyncScope): Observable<{ job_id: string }> {
    return this.http.post<{ job_id: string }>(`${BASE}/sync/full`, { confirm: true, scope });
  }

  startIncremental(): Observable<{ job_id: string }> {
    return this.http.post<{ job_id: string }>(`${BASE}/sync/incremental`, {});
  }

  history(limit = 50): Observable<{ jobs: Job[] }> {
    return this.http.get<{ jobs: Job[] }>(`${BASE}/sync/history`, { params: { limit } });
  }

  failures(limit = 200): Observable<{ failures: FailedImage[] }> {
    return this.http.get<{ failures: FailedImage[] }>(`${BASE}/sync/failures`, { params: { limit } });
  }

  retryFailed(): Observable<{ job_id: string; requeued_images: number }> {
    return this.http.post<{ job_id: string; requeued_images: number }>(`${BASE}/sync/retry-failed`, {});
  }
}
