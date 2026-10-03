import { DecimalPipe } from '@angular/common';
import { Component, computed, DestroyRef, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject, catchError, interval, map, merge, of, startWith, switchMap } from 'rxjs';

import { Api, errorMessage } from '../../core/api';
import { SyncStatus } from '../../core/models';
import { TokenBox } from '../../shared/token-box';

@Component({
  selector: 'app-dashboard',
  imports: [DecimalPipe, TokenBox],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.scss',
})
export class DashboardPage {
  private readonly api = inject(Api);
  private readonly refresh$ = new Subject<void>();

  protected readonly status = signal<SyncStatus | null>(null);
  protected readonly error = signal<string | null>(null);
  protected readonly needsToken = signal(false);
  protected readonly loading = signal(true);

  protected readonly indexedPct = computed(() => {
    const s = this.status();
    const total = s?.sql_server.images ?? 0;
    return total ? Math.min(100, (s!.images.indexed / total) * 100) : 0;
  });

  constructor() {
    merge(interval(5000), this.refresh$).pipe(
      startWith(0),
      switchMap(() => this.api.status().pipe(map((s) => ({ s, e: null as unknown })), catchError((e) => of({ s: null, e })))),
      takeUntilDestroyed(inject(DestroyRef)),
    ).subscribe(({ s, e }) => {
      this.loading.set(false);
      if (s) {
        this.status.set(s);
        this.error.set(null);
        this.needsToken.set(false);
      } else {
        const msg = errorMessage(e);
        this.needsToken.set(/^(401|503):/.test(msg));
        this.error.set(msg);
      }
    });
  }

  protected reload(): void {
    this.refresh$.next();
  }

  protected fmt(v: number | null | undefined): string {
    return typeof v === 'number' ? v.toLocaleString() : 'n/a';
  }

  protected when(iso: string | null | undefined): string {
    return iso ? iso.slice(0, 16).replace('T', ' ') + ' UTC' : 'Never';
  }
}
