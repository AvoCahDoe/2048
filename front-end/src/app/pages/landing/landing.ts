import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-landing',
  standalone: true,
  imports: [RouterLink],
  templateUrl: './landing.html',
  styleUrl: './landing.scss',
})
export class LandingPage {
  tiles = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 2, 4, 8, 16, 32];

  tileBg(t: number): string {
    const map: Record<number, string> = {
      2: '#1a2230',
      4: '#1e2a3a',
      8: '#243447',
      16: '#2c3f52',
      32: '#3a4f3a',
      64: '#4a5c2e',
      128: '#5c4a1e',
      256: '#6b3f18',
      512: '#7a3518',
      1024: '#8a2a20',
      2048: '#a61e18',
    };
    return map[t] ?? '#1a2230';
  }
}
