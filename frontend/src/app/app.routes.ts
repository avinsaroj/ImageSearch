import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'search' },
  { path: 'search', title: 'Search · Jewellery Search', loadComponent: () => import('./pages/search/search').then((m) => m.SearchPage) },
  { path: 'dashboard', title: 'Dashboard · Jewellery Search', loadComponent: () => import('./pages/dashboard/dashboard').then((m) => m.DashboardPage) },
  { path: 'sync', title: 'Sync admin · Jewellery Search', loadComponent: () => import('./pages/sync/sync').then((m) => m.SyncPage) },
  { path: '**', redirectTo: 'search' },
];
