import { DecimalPipe } from '@angular/common';
import { Component, computed, effect, HostListener, inject, signal } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { catchError, of } from 'rxjs';

import { Api, errorMessage } from '../../core/api';
import { ProductResult, RankMode, SearchParams, SearchResponse, Tri } from '../../core/models';

const DEFAULTS: SearchParams = {
  top_k: 12, mode: 'fusion', category: '', plain_gold: 'all', solitaire: 'all', valid: 'all',
  franchise: 'all', item_codes: '', min_score: 0,
};

@Component({
  selector: 'app-search',
  imports: [DecimalPipe],
  templateUrl: './search.html',
  styleUrl: './search.scss',
})
export class SearchPage {
  private readonly api = inject(Api);

  protected readonly tri: { value: Tri; label: string }[] = [
    { value: 'all', label: 'All' }, { value: 'yes', label: 'Yes' }, { value: 'no', label: 'No' },
  ];
  protected readonly flags = [
    { key: 'plain_gold', label: 'Plain gold' }, { key: 'solitaire', label: 'Solitaire' },
    { key: 'valid', label: 'Valid' }, { key: 'franchise', label: 'Franchise item' },
  ] as const;
  protected readonly modes: { value: RankMode; label: string }[] = [
    { value: 'fusion', label: 'CLIP + DINOv2' }, { value: 'clip', label: 'CLIP only' }, { value: 'dino', label: 'DINOv2 only' },
  ];

  protected readonly categories = toSignal(this.api.categories().pipe(catchError(() => of(null))), { initialValue: null });
  protected readonly params = signal<SearchParams>({ ...DEFAULTS });
  protected readonly file = signal<File | null>(null);
  protected readonly preview = signal<string | null>(null);
  protected readonly dragging = signal(false);
  protected readonly loading = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly result = signal<SearchResponse | null>(null);
  protected readonly selected = signal<ProductResult | null>(null);
  protected readonly filtersOpen = signal(false);

  protected readonly activeFilters = computed(() => {
    const p = this.params();
    return [p.category, p.plain_gold, p.solitaire, p.valid, p.franchise, p.item_codes.trim(), p.min_score > 0]
      .filter((v) => v && v !== 'all').length;
  });

  constructor() {
    // Revoke the object URL of the previous preview so large uploads do not pile up in memory.
    effect((onCleanup) => {
      const url = this.preview();
      onCleanup(() => url && URL.revokeObjectURL(url));
    });
  }

  protected set<K extends keyof SearchParams>(key: K, value: SearchParams[K]): void {
    this.params.update((p) => ({ ...p, [key]: value }));
  }

  protected reset(): void {
    this.params.set({ ...DEFAULTS });
  }

  protected onPick(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files?.[0]) this.setFile(input.files[0]);
    input.value = '';
  }

  protected onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(false);
    const f = event.dataTransfer?.files?.[0];
    if (f) this.setFile(f);
  }

  @HostListener('window:paste', ['$event'])
  protected onPaste(event: ClipboardEvent): void {
    const f = Array.from(event.clipboardData?.files ?? []).find((x) => x.type.startsWith('image/'));
    if (f) this.setFile(f);
  }

  @HostListener('document:keydown.escape')
  protected closeDialog(): void {
    this.selected.set(null);
  }

  private setFile(f: File): void {
    if (!f.type.startsWith('image/')) {
      this.error.set('Please choose an image file (JPG, PNG, WEBP).');
      return;
    }
    this.file.set(f);
    this.preview.set(URL.createObjectURL(f));
    this.error.set(null);
    this.result.set(null);
  }

  protected clear(): void {
    this.file.set(null);
    this.preview.set(null);
    this.result.set(null);
    this.error.set(null);
  }

  protected search(): void {
    const f = this.file();
    if (!f || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    this.api.searchImage(f, this.params()).subscribe({
      next: (res) => {
        this.result.set(res);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(errorMessage(err));
        this.result.set(null);
        this.loading.set(false);
      },
    });
  }

  protected chipClass(v: boolean | null): string {
    return v === true ? 'yes' : v === false ? 'no' : '';
  }

  protected spaces(r: ProductResult): { name: string; value: number }[] {
    return Object.entries(r.score_per_space ?? {}).map(([name, value]) => ({ name, value }));
  }

  protected hideBroken(event: Event): void {
    (event.target as HTMLImageElement).style.visibility = 'hidden';
  }
}
