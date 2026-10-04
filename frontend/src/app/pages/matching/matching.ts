import { DecimalPipe } from '@angular/common';
import { Component, computed, effect, HostListener, inject, signal } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { catchError, of } from 'rxjs';
import { Api, errorMessage } from '../../core/api';
import { CrossCategoryResponse, ProductResult, RankMode, Tri } from '../../core/models';

/** Categories ticked by default once the real category codes load (matched by name, case-insensitive). */
const DEFAULT_TARGETS = ['EARRING', 'RING', 'BRACELET', 'BANGLE', 'PENDANT'];

@Component({
  selector: 'app-matching',
  imports: [DecimalPipe],
  templateUrl: './matching.html',
  styleUrl: './matching.scss',
})
export class MatchingPage {
  private readonly api = inject(Api);

  protected readonly modes: { value: RankMode; label: string }[] = [
    { value: 'fusion', label: 'CLIP + DINOv2' },
    { value: 'clip', label: 'CLIP only' },
    { value: 'dino', label: 'DINOv2 only' },
  ];
  protected readonly tri: { value: Tri; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'yes', label: 'Valid only' },
  ];

  protected readonly categories = toSignal(this.api.categories().pipe(catchError(() => of(null))));
  protected readonly file = signal<File | null>(null);
  protected readonly preview = signal<string | null>(null);
  protected readonly dragging = signal(false);
  protected readonly source = signal('');
  protected readonly targets = signal<string[]>([]);
  protected readonly perCategory = signal(6);
  protected readonly mode = signal<RankMode>('fusion');
  protected readonly valid = signal<Tri>('yes');
  protected readonly loading = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly result = signal<CrossCategoryResponse | null>(null);
  protected readonly selected = signal<ProductResult | null>(null);

  /** Every category except the one the uploaded product belongs to. */
  protected readonly choices = computed(() =>
    (this.categories()?.categories ?? []).map((c) => c.code).filter((c) => c !== this.source()),
  );
  protected readonly shown = computed(() => (this.result()?.groups ?? []).filter((g) => g.results.length));
  protected readonly empty = computed(() => (this.result()?.groups ?? []).filter((g) => !g.results.length));
  protected readonly emptyNames = computed(() => this.empty().map((g) => g.category).join(', '));

  constructor() {
    effect((onCleanup) => {
      const url = this.preview();
      onCleanup(() => url && URL.revokeObjectURL(url));
    });
    // preselect the usual companion categories once the list is known
    effect(() => {
      const codes = (this.categories()?.categories ?? []).map((c) => c.code);
      if (codes.length && !this.targets().length) {
        this.targets.set(codes.filter((c) => DEFAULT_TARGETS.some((k) => c.toUpperCase().includes(k))));
      }
    });
  }

  protected setSource(code: string): void {
    this.source.set(code);
    this.targets.update((t) => t.filter((c) => c !== code)); // a design is never matched against its own category
  }

  protected toggle(code: string): void {
    this.targets.update((t) => (t.includes(code) ? t.filter((c) => c !== code) : [...t, code]));
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
    if (!f || !this.targets().length || this.loading()) return;
    this.loading.set(true);
    this.error.set(null);
    this.api.crossCategory(f, this.targets(), this.perCategory(), this.mode(), this.valid()).subscribe({
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

  protected hideBroken(e: Event): void {
    (e.target as HTMLImageElement).style.visibility = 'hidden';
  }
}
