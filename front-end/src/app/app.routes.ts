import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () =>
      import('./pages/landing/landing').then((m) => m.LandingPage),
  },
  {
    path: 'docs',
    loadComponent: () => import('./pages/docs/docs').then((m) => m.DocsPage),
  },
  {
    path: 'try',
    loadComponent: () => import('./pages/try/try').then((m) => m.TryPage),
  },
  {
    path: 'results',
    loadComponent: () =>
      import('./pages/results/results').then((m) => m.ResultsPage),
  },
  {
    path: 'play',
    loadComponent: () => import('./game/game').then((m) => m.GameComponent),
  },
  { path: '**', redirectTo: '' },
];
