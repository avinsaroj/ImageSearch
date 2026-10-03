import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'search' },
  { path: 'search', title: 'Search · Kisna Image Search', loadComponent: () => import('./pages/search/search').then((m) => m.SearchPage) },
  { path: 'dashboard', title: 'Dashboard · Kisna Image Search', loadComponent: () => import('./pages/dashboard/dashboard').then((m) => m.DashboardPage) },
  { path: 'sync', title: 'Sync admin · Kisna Image Search', loadComponent: () => import('./pages/sync/sync').then((m) => m.SyncPage) },
  { path: '**', redirectTo: 'search' },
];
