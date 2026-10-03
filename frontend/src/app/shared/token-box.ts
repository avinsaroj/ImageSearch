import { Component, inject, output } from '@angular/core';

import { AuthService } from '../core/auth';

/** Inline admin-token prompt shown by admin pages when the API rejects the request. */
@Component({
  selector: 'app-token-box',
  template: `
    <div class="card box">
      <div>
        <h3>Admin token required</h3>
        <p class="muted small">Enter the <span class="mono">ADMIN_API_TOKEN</span> from the server's .env. It is kept for this browser tab only.</p>
      </div>
      <form (submit)="$event.preventDefault(); save(input.value)">
        <input #input type="password" placeholder="X-Admin-Token" aria-label="Admin token" [value]="auth.token()" />
        <button class="btn primary" type="submit">Save</button>
      </form>
    </div>
  `,
  styles: `
    .box { display: grid; gap: 12px; max-width: 520px; }
    form { display: flex; gap: 8px; }
  `,
})
export class TokenBox {
  protected readonly auth = inject(AuthService);
  readonly saved = output<void>();

  protected save(value: string): void {
    this.auth.set(value);
    this.saved.emit();
  }
}
