import { Component, inject, signal } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { interval, startWith, switchMap } from 'rxjs';
import { toSignal } from '@angular/core/rxjs-interop';

import { Api } from './core/api';
import { AuthService } from './core/auth';
import { Icon, IconName } from './shared/icon';

interface NavGroup {
  title: string;
  items: { path: string; label: string; icon: IconName }[];
}

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive, Icon],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  private readonly api = inject(Api);
  protected readonly auth = inject(AuthService);

  protected readonly nav: NavGroup[] = [
    {
      title: 'Discover',
      items: [
        { path: '/search', label: 'Search', icon: 'search' },
        { path: '/matching-set', label: 'Matching set', icon: 'layers' },
      ],
    },
    {
      title: 'Admin',
      items: [
        { path: '/dashboard', label: 'Dashboard', icon: 'chart' },
        { path: '/sync', label: 'Sync admin', icon: 'refresh' },
      ],
    },
  ];
  protected readonly online = toSignal<boolean | null>(
    interval(15000).pipe(startWith(0), switchMap(() => this.api.online())),
    { initialValue: null },
  );
  protected readonly tokenOpen = signal(false);
}
