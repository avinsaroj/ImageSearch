import { Component, inject, signal } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { interval, startWith, switchMap } from 'rxjs';
import { toSignal } from '@angular/core/rxjs-interop';

import { Api } from './core/api';
import { AuthService } from './core/auth';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  private readonly api = inject(Api);
  protected readonly auth = inject(AuthService);

  protected readonly nav = [
    { path: '/search', label: 'Search', icon: '🔍' },
    { path: '/dashboard', label: 'Dashboard', icon: '📊' },
    { path: '/sync', label: 'Sync admin', icon: '🔄' },
  ];
  protected readonly online = toSignal<boolean | null>(
    interval(15000).pipe(startWith(0), switchMap(() => this.api.online())),
    { initialValue: null },
  );
  protected readonly tokenOpen = signal(false);
}
