import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  links = [
    { path: '/', label: 'Lab', exact: true },
    { path: '/docs', label: 'Docs', exact: false },
    { path: '/try', label: 'Try', exact: false },
    { path: '/results', label: 'Results', exact: false },
  ];
}
