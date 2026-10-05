import { DecimalPipe } from '@angular/common';
import { Component, computed, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed, toSignal } from '@angular/core/rxjs-interop';
import { Observable, Subject, catchError, interval, map, merge, of, startWith, switchMap } from 'rxjs';

import { Api, errorMessage } from '../../core/api';
import { FailedImage, Job, ScopeList, ScopePreview, SyncScope, SyncStatus } from '../../core/models';
import { Icon } from '../../shared/icon';
import { TokenBox } from '../../shared/token-box';

type Tab = 'failed' | 'history';
type Busy = 'preview' | 'add' | 'full' | 'incremental' | 'retry' | 'cancel';
type Flag = 'plain_gold' | 'solitaire' | 'valid' | 'franchise';

const EMPTY_SCOPE: SyncScope = {
  categories: [], plain_gold: null, solitaire: null, valid: null, franchise: null, search_text: null, max_products: null,
};

@Component({
  selector: 'app-sync',
  imports: [DecimalPipe, Icon, TokenBox],
  templateUrl: './sync.html',
  styleUrl: './sync.scss',
})
export class SyncPage {
  private readonly api = inject(Api);
  private readonly refresh$ = new Subject<void>();

  protected readonly flags: { key: Flag; label: string }[] = [
    { key: 'plain_gold', label: 'Plain gold' }, { key: 'solitaire', label: 'Solitaire' },
    { key: 'valid', label: 'Valid' }, { key: 'franchise', label: 'Franchise item' },
  ];
  protected readonly triOptions: { value: boolean | null; label: string }[] = [
    { value: null, label: 'Any' }, { value: true, label: 'Yes' }, { value: false, label: 'No' },
  ];

  protected readonly categories = toSignal(this.api.categories().pipe(catchError(() => of(null))), { initialValue: null });
  protected readonly status = signal<SyncStatus | null>(null);
  protected readonly needsToken = signal(false);
  protected readonly statusError = signal<string | null>(null);

  protected readonly scope = signal<SyncScope>({ ...EMPTY_SCOPE });
  protected readonly confirm = signal(false);
  protected readonly preview = signal<ScopePreview | null>(null);
  protected readonly busy = signal<Busy | null>(null);
  protected readonly notice = signal<{ kind: 'ok' | 'error'; text: string } | null>(null);

  protected readonly tab = signal<Tab>('failed');
  protected readonly failures = signal<FailedImage[] | null>(null);
  protected readonly history = signal<Job[] | null>(null);
  protected readonly lastScope = signal<ScopeList | null>(null);

  protected readonly job = computed(() => this.status()?.current_job ?? null);
  protected readonly progress = computed(() => {
    const c = this.job()?.counters;
    const total = c?.to_process ?? 0;
    return total ? Math.min(100, ((c?.processed ?? 0) / total) * 100) : 0;
  });
  protected readonly scopeList = computed(() => (this.lastScope()?.scopes ?? []).map((s) => this.describe(s)));

  constructor() {
    merge(interval(3000), this.refresh$).pipe(
      startWith(0),
      switchMap(() => this.api.status().pipe(map((s) => ({ s, e: null as unknown })), catchError((e) => of({ s: null, e })))),
      takeUntilDestroyed(inject(DestroyRef)),
    ).subscribe(({ s, e }) => {
      if (s) {
        const finished = this.job() && !s.current_job;
        this.status.set(s);
        this.statusError.set(null);
        this.needsToken.set(false);
        if (finished) this.loadTabs();
      } else {
        const msg = errorMessage(e);
        this.needsToken.set(/^(401|503):/.test(msg));
        this.statusError.set(msg);
      }
    });
    this.loadTabs();
  }

  protected retryAuth(): void {
    this.refresh$.next();
    this.loadTabs();
  }

  private loadTabs(): void {
    this.api.failures().subscribe({ next: (r) => this.failures.set(r.failures), error: () => this.failures.set(null) });
    this.api.history().subscribe({ next: (r) => this.history.set(r.jobs), error: () => this.history.set(null) });
    this.api.scope().subscribe({ next: (r) => this.lastScope.set(r), error: () => this.lastScope.set(null) });
  }

  // scope form
  protected toggleCategory(code: string): void {
    this.scope.update((s) => ({
      ...s, categories: s.categories.includes(code) ? s.categories.filter((c) => c !== code) : [...s.categories, code],
    }));
    this.preview.set(null);
  }

  protected setFlag(key: Flag, value: boolean | null): void {
    this.scope.update((s) => ({ ...s, [key]: value }));
    this.preview.set(null);
  }

  protected setText(value: string): void {
    this.scope.update((s) => ({ ...s, search_text: value.trim() || null }));
    this.preview.set(null);
  }

  protected setLimit(value: string): void {
    const n = Math.floor(Number(value));
    this.scope.update((s) => ({ ...s, max_products: n > 0 ? n : null }));
    this.preview.set(null);
  }

  protected clearScope(): void {
    this.scope.set({ ...EMPTY_SCOPE });
    this.preview.set(null);
  }

  // actions
  protected runPreview(): void {
    this.run('preview', this.api.preview(this.scope()), (p) => this.preview.set(p));
  }

  protected startAdd(): void {
    this.run('add', this.api.startAdd(this.scope()), (r) =>
      this.notice.set({ kind: 'ok', text: `Adding to the index (job ${r.job_id.slice(0, 8)}). Search keeps working while it runs.` }));
  }

  protected startFull(): void {
    this.run('full', this.api.startFull(this.scope()), (r) => {
      this.notice.set({ kind: 'ok', text: `Rebuild queued (job ${r.job_id.slice(0, 8)}).` });
      this.confirm.set(false);
    });
  }

  protected cancelJob(): void {
    this.run('cancel', this.api.cancel(), (r) =>
      this.notice.set({ kind: 'ok', text: r.status === 'cancelling' ? 'Stopping… the job ends at its next checkpoint.' : 'Job cancelled.' }));
  }

  protected startIncremental(): void {
    this.run('incremental', this.api.startIncremental(), (r) =>
      this.notice.set({ kind: 'ok', text: `Incremental sync queued (job ${r.job_id.slice(0, 8)}).` }));
  }

  protected retryFailed(): void {
    this.run('retry', this.api.retryFailed(), (r) =>
      this.notice.set({ kind: 'ok', text: `Re-queued ${r.requeued_images} images (job ${r.job_id.slice(0, 8)}).` }));
  }

  private run<T>(busy: Busy, call: Observable<T>, ok: (v: T) => void): void {
    if (this.busy()) return;
    this.busy.set(busy);
    this.notice.set(null);
    call.subscribe({
      next: (v) => { ok(v); this.busy.set(null); this.refresh$.next(); this.loadTabs(); },
      error: (e) => { this.notice.set({ kind: 'error', text: errorMessage(e) }); this.busy.set(null); },
    });
  }

  protected describe(s: Partial<SyncScope> | null): string {
    if (!s) return 'everything';
    const yn = (v: boolean | null | undefined) => (v === true ? 'yes' : 'no');
    const parts: string[] = [];
    if (s.categories?.length) parts.push(`categories: ${s.categories.join(', ')}`);
    for (const f of this.flags) if (s[f.key] != null) parts.push(`${f.label.toLowerCase()}: ${yn(s[f.key])}`);
    if (s.search_text) parts.push(`text “${s.search_text}”`);
    if (s.max_products) parts.push(`first ${s.max_products} products`);
    const extra = s as { item_id_max?: number | null };
    if (extra.item_id_max) parts.push(`ItemID ≤ ${extra.item_id_max}`);
    return parts.join(' · ') || 'everything';
  }

  protected when(iso: string | null): string {
    return iso ? iso.slice(0, 19).replace('T', ' ') : '–';
  }

  protected statusClass(s: string): string {
    return s === 'succeeded' ? 'yes' : s === 'failed' ? 'bad' : s === 'running' || s === 'queued' ? 'warn' : '';
  }
}
