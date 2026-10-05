import { DecimalPipe } from '@angular/common';
import { Component, input, output } from '@angular/core';

import { ProductResult, SimilarResponse } from '../core/models';
import { Icon } from './icon';

/** Product detail modal shared by the Search and Matching set pages. */
@Component({
  selector: 'app-product-dialog',
  imports: [DecimalPipe, Icon],
  template: `
    @let r = product();
    <div class="backdrop" (click)="closed.emit()">
      <div class="dialog" role="dialog" aria-modal="true" [attr.aria-label]="'Product ' + r.item_code" (click)="$event.stopPropagation()">
        <button class="btn ghost icon-btn close" type="button" aria-label="Close" (click)="closed.emit()"><app-icon name="x" /></button>
        <div class="d-main">
          <img class="d-img" [src]="r.best_image.ImageURL" [alt]="r.item_code" />
          <div class="d-info">
            <div>
              <div class="eyebrow">{{ r.category_code }}</div>
              <h2>{{ r.item_code }}</h2>
            </div>
            <div class="chips">
              <span [class]="'chip ' + chip(r.is_plain_gold)">Plain gold</span>
              <span [class]="'chip ' + chip(r.is_solitaire)">Solitaire</span>
              <span [class]="'chip ' + chip(r.is_valid)">Valid</span>
              <span [class]="'chip ' + chip(r.is_franchise_item)">Franchise</span>
            </div>
            <dl class="facts">
              <dt>ItemID</dt><dd class="num">{{ r.item_id }}</dd>
              <dt>Status remark</dt><dd>{{ r.status_remark || '–' }}</dd>
              <dt>Similarity</dt><dd class="num">{{ r.score | number: '1.4-4' }}</dd>
            </dl>
            <div class="d-actions">
              <button class="btn primary" type="button" [disabled]="loading()" (click)="findSimilar.emit()">
                @if (loading()) { <span class="spinner"></span> Finding… } @else { <app-icon name="sparkles" /> Find similar products }
              </button>
              <a class="btn" [href]="r.best_image.ImageURL" target="_blank" rel="noopener"><app-icon name="external" /> Original image</a>
            </div>
          </div>
        </div>

        @if (error(); as e) { <div class="alert error" role="alert"><app-icon name="alert" /> {{ e }}</div> }
        @if (similar(); as sim) {
          <h3>Similar to {{ r.item_code }} ({{ sim.total_returned }})</h3>
          @if (!sim.results.length) {
            <div class="alert">No similar products matched. Try lowering the minimum similarity or clearing filters.</div>
          } @else {
            <div class="gallery clickable">
              @for (s of sim.results; track s.item_id) {
                <figure tabindex="0" (click)="picked.emit(s)" (keydown.enter)="picked.emit(s)">
                  <img [src]="s.best_image.ImageURL" [alt]="s.item_code" loading="lazy" (error)="hide($event)" />
                  <figcaption><strong>{{ s.item_code }}</strong> · {{ s.score | number: '1.3-3' }}</figcaption>
                </figure>
              }
            </div>
          }
        }

        @if (r.images.length > 1) {
          <h3>All images ({{ r.images.length }})</h3>
          <div class="gallery">
            @for (img of r.images; track img.ImgID) {
              <figure>
                <img [src]="img.ImageURL" [alt]="'Image ' + img.ImgID" loading="lazy" (error)="hide($event)" />
                <figcaption>#{{ img.ImgID }} · view {{ img.ImgViewID }}{{ img.ImgID === r.best_image.ImgID ? ' ★' : '' }}</figcaption>
              </figure>
            }
          </div>
        }
      </div>
    </div>
  `,
})
export class ProductDialog {
  readonly product = input.required<ProductResult>();
  readonly similar = input<SimilarResponse | null>(null);
  readonly loading = input(false);
  readonly error = input<string | null>(null);
  readonly closed = output<void>();
  readonly findSimilar = output<void>();
  readonly picked = output<ProductResult>();

  protected chip(v: boolean | null): string {
    return v === true ? 'yes' : '';
  }

  protected hide(event: Event): void {
    (event.target as HTMLImageElement).style.visibility = 'hidden';
  }
}
